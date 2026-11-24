"""Exercise the unified Qt screens and the real organizer renderer under Xvfb."""
from pathlib import Path
import json
import time
from PyQt5 import QtCore, QtWidgets, QtTest


def attach(window, arguments):
    output, forward_data, inverse_data, forward_result, inverse_result = map(Path,arguments[:5])
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'running'}
    started = time.monotonic()
    phase = ['home']
    logged_phase = [None]
    (output/'ui-check.json').write_text(json.dumps(report))
    import signal
    signal.signal(signal.SIGTERM, lambda *_: finish(RuntimeError('SIGTERM during '+phase[0])))
    page = [None]
    def save(name):
        QtWidgets.QApplication.processEvents()
        window.grab().save(str(output / name))
    def finish(error=None):
        report.update(status='failed' if error else 'passed',elapsed_s=time.monotonic()-started)
        if error:report['error']=repr(error)
        (output/'ui-check.json').write_text(json.dumps(report,indent=2))
        timer.stop()
        for p in window.pages.values():
            if hasattr(p,'stop'):p.stop()
        QtWidgets.QApplication.exit(1 if error else 0)
    def tick():
        try:
            if time.monotonic()-started>600:raise RuntimeError('UI check timed out: '+phase[0])
            state=phase[0]
            if logged_phase[0] != state:
                print(f'UI phase {state}; elapsed {time.monotonic()-started:.1f}s', flush=True)
                logged_phase[0]=state
            if state=='home':
                save('home.png');window.open_page('forward');page[0]=window.pages['forward']
                assert window.title_bar.brand.text() == 'Forward solver'
                page[0].add_directory(inverse_data)
                page[0].add_directory(forward_data)
                assert page[0].source_choice.count() == 2
                page[0].source_choice.setCurrentIndex(0)
                assert page[0].active == inverse_data.resolve()
                page[0].source_choice.setCurrentIndex(1)
                assert page[0].tree.currentItem() is page[0].sources[forward_data.resolve()]
                page[0].run_name.setText('forward-ui-' + str(time.time_ns()))
                item=page[0].tree.currentItem()
                assert item.childCount()==6 and not item.isExpanded()
                rect=page[0].tree.visualItemRect(item)
                QtTest.QTest.mouseClick(page[0].tree.viewport(),QtCore.Qt.LeftButton,pos=rect.center())
                assert item.isExpanded()
                assert len(page[0].editors)==12
                save('forward-main.png')
                if '--portal' in arguments:
                    page[0].pick_button.click()
                    def accept_portal():
                        import subprocess
                        subprocess.run(['xdotool','search','--onlyvisible','--name','Pick a folder with data',
                                        'windowfocus','key','Return'],check=True,timeout=5)
                    QtCore.QTimer.singleShot(1800, accept_portal)
                    phase[0]='portal'
                else:
                    page[0].start_button.click();assert page[0].running and not page[0].table.isEnabled()
                    QtCore.QTimer.singleShot(800,page[0].stop_button.click)
                    phase[0]='forward-stop'
            elif state=='portal':
                if page[0].picker_task is not None:return
                assert page[0].active == forward_data.resolve()
                report['native_portal_checked']=True
                page[0].start_button.click();assert page[0].running
                QtCore.QTimer.singleShot(800,page[0].stop_button.click)
                phase[0]='forward-stop'
            elif state=='forward-stop':
                if page[0].running:return
                assert json.loads((page[0].runner.directory/'status.json').read_text())['status']=='cancelled'
                report['forward_stop_checked']=True
                assert page[0].runner.directory.name == f'{forward_data.name}_run-{page[0].run_name.text()}'
                page[0].add_result(forward_result)
                page[0].tabs.setCurrentIndex(1);phase[0]='forward-view'
            elif state=='forward-view':
                if page[0].viewer.task is not None:return
                viewer=page[0].viewer.viewer
                assert viewer is not None, page[0].log.toPlainText()
                assert 'Temperature' in viewer.properties
                assert viewer._active_layers
                viewer.time_index=len(viewer.current_times)-1;viewer._update_buttons();viewer._draw()
                viewer.profile_mode.setCurrentIndex(1)
                assert len(viewer.plot_widget.items())>0
                save('forward-visualize.png')
                report['forward_viewer_properties']=viewer.properties
                window.title_bar.back.click();assert window.stack.currentWidget() is window.home
                window.open_page('inverse');page[0]=window.pages['inverse']
                page[0].add_directory(inverse_data)
                assert page[0].tree.topLevelItem(0).childCount()>=7
                assert len(page[0].editors)==16
                save('inverse-main.png')
                page[0].start_button.click();assert page[0].running
                QtCore.QTimer.singleShot(800,page[0].stop_button.click)
                phase[0]='inverse-stop'
            elif state=='inverse-stop':
                if page[0].running:return
                assert json.loads((page[0].runner.directory/'status.json').read_text())['status']=='cancelled'
                report['inverse_stop_checked']=True
                page[0].add_result(inverse_result)
                page[0].tabs.setCurrentIndex(1);phase[0]='inverse-view'
            elif state=='inverse-view':
                if page[0].viewer.task is not None:return
                viewer=page[0].viewer.viewer;assert viewer is not None
                for prop in ('Temperature','Sweeney_Burnham_1990_EASY_Ro','Heat_Flow_Basal'):
                    assert prop in viewer.properties
                viewer.time_index=len(viewer.current_times)-1;viewer._update_buttons();viewer._draw()
                save('inverse-simple.png')
                page[0].advanced_button.click()
                assert page[0].views.currentIndex()==1 and not page[0].advanced.listDataItems()
                assert page[0].advanced.getPlotItem().getAxis('left').labelText=='q'
                before=page[0].advanced.viewRange()
                QtTest.QTest.mouseClick(page[0].advanced.viewport(),QtCore.Qt.LeftButton)
                assert page[0].advanced.viewRange()==before
                save('inverse-advanced.png')
                report['advanced_inert_empty_axes']=True
                page[0].simple_button.click();assert page[0].views.currentIndex()==0
                window.title_bar.back.click();window.open_page('viewer');page[0]=window.pages['viewer']
                assert window.title_bar.brand.text() == 'Visualization'
                assert page[0].choice.count() == 2
                page[0].choice.setCurrentIndex(page[0].choice.findData(str(forward_result)))
                phase[0]='standalone'
            elif state=='standalone':
                if page[0].viewer.task is not None:return
                assert page[0].viewer.viewer is not None
                save('visualize-data.png')
                report['organizer_renderer_embedded']=True
                report['tree_input_files_checked']=True
                finish()
        except Exception as error:finish(error)
    timer=QtCore.QTimer(window);timer.timeout.connect(tick);timer.start(150)
