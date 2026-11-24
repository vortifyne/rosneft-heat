"""
    System direcotry choose via XDG FileChooser portal.

    D-Bus client is bundled
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import queue
import sys
import threading
from urllib.parse import unquote_to_bytes, urlsplit
import uuid


class PickerError(RuntimeError):
    pass


def initial_directory(value: str) -> Path:
    path = Path(value).expanduser().absolute() if value.strip() else Path.home()

    while not path.is_dir() and path != path.parent:
        path = path.parent

    return path


def local_directory(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment:
        raise PickerError('Выберите локальный каталог.')

    path = Path(os.fsdecode(unquote_to_bytes(parsed.path)))
    if not path.is_absolute() or not path.is_dir():
        raise PickerError('Выбранный каталог недоступен.')

    return str(path)


async def portal_directory(parent: str, title: str, current: str, cancelled=lambda: False) -> str | None:
    from dbus_next import Message, MessageType, Variant
    from dbus_next.aio import MessageBus

    bus = await asyncio.wait_for(MessageBus().connect(), 10)
    service = 'org.freedesktop.portal.Desktop'
    responses = {}
    arrived = asyncio.Event()

    def receive(message):
        if (message.message_type == MessageType.SIGNAL and
                message.interface == 'org.freedesktop.portal.Request' and message.member == 'Response'):
            responses[message.path] = message.body
            arrived.set()

    try:
        bus.add_message_handler(receive)
        reply = await asyncio.wait_for(bus.call(Message(
            destination='org.freedesktop.DBus', path='/org/freedesktop/DBus',
            interface='org.freedesktop.DBus', member='AddMatch', signature='s',
            body=["type='signal',sender='org.freedesktop.portal.Desktop',interface='org.freedesktop.portal.Request',member='Response'"])), 10)

        if reply.message_type == MessageType.ERROR:
            raise PickerError(reply.body[0])

        options = {
            'handle_token': Variant('s', 'heat_' + uuid.uuid4().hex),
            'directory': Variant('b', True), 'multiple': Variant('b', False),
            'modal': Variant('b', True), 'accept_label': Variant('s', 'Выбрать папку'),
            'current_folder': Variant('ay', os.fsencode(initial_directory(current)) + b'\0'),
        }
        reply = await asyncio.wait_for(bus.call(Message(
            destination=service, path='/org/freedesktop/portal/desktop',
            interface='org.freedesktop.portal.FileChooser', member='OpenFile',
            signature='ssa{sv}', body=[parent, title, options])), 15)

        if reply.message_type == MessageType.ERROR:
            raise PickerError(reply.body[0])

        handle = reply.body[0]
        disconnected = asyncio.create_task(bus.wait_for_disconnect())
        try:
            while handle not in responses:
                if cancelled():
                    await asyncio.wait_for(bus.call(Message(destination=service, path=handle,
                        interface='org.freedesktop.portal.Request', member='Close')), 2)
                    return None
                arrived.clear()
                signal = asyncio.create_task(arrived.wait())
                done, _ = await asyncio.wait((signal, disconnected), timeout=.1, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    signal.cancel()
                if disconnected in done:
                    signal.cancel()
                    raise PickerError('Системный диалог закрыл соединение.')
            code, results = responses[handle]
        finally:
            disconnected.cancel()

        if code in (1, 2):
            return None
        if code != 0:
            raise PickerError('Системный диалог не смог выбрать каталог.')
        uris = results.get('uris')
        if uris is None or len(uris.value) != 1:
            raise PickerError('Системный диалог не вернул каталог.')
        return local_directory(uris.value[0])
    finally:
        bus.disconnect()

