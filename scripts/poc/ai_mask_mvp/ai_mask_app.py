#!/usr/bin/env python3
"""Desktop UI of the AI Mask MVP (issue #5): one photo, the selected model set, and the mask tools.

Input:  a photo (JPEG, PNG, or TIFF; scaled to a long edge of at most 4096 px) and the ONNX graphs
        of mask_pipeline.GraphPaths (under build/tmp/explore/ by default).
Output: the mask as an 8-bit PNG at the photo size, the prompts as JSON, and with --log a JSON
        record of the stage times and the graph SHA-256.

Tools:

  Subject, Background   BiRefNet lite 1024 (F1, F2); Background is the inverted Subject alpha
  Sky, People           head run 2 + SAM 2.1 tiny decoder (F3, F4a); the People list selects one
                        person or all people; "not found" is a message, not an empty mask
  Stroke tool           drag a stroke: 4 points at equal spacing, one decoder run when the stroke
                        ends; a click (no drag) is one point (F5). Alt subtracts (F6). All points
                        of the selection go into one decoder run with the previous mask
  Box tool              drag a box; it replaces the box of the selection (F5)
  [ and ]               smaller or larger mask of a click (the three multimask outputs, no model run)
  Feather, Edge         blur and dilate or erode of the final mask (F8)
  Overlay               color, opacity; Y shows or hides the overlay (F10)

All models load together when the program starts. The model calls run one at a time in one worker
thread; the UI does not accept a new model call while one runs. A model error (for example the
DirectML error 887A0006) is shown as it is; the program does not change the provider.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot  # noqa: E402
from PySide6.QtGui import (QAction, QColor, QImage, QKeySequence, QPainter, QPainterPath,  # noqa: E402
                           QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QColorDialog,  # noqa: E402
                               QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QRadioButton, QSlider, QVBoxLayout,
                               QWidget)

DISPLAY_MAX_EDGE = 2048
CLICK_DRAG_LIMIT_SCREEN_PX = 3.0
OVERLAY_UPDATE_DELAY_MS = 120
EDGE_RANGE_PX = 50
FEATHER_RANGE_PX = 50


def rgb_qimage(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array)
    height, width, _ = array.shape
    return QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def rgba_qimage(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array)
    height, width, _ = array.shape
    return QImage(array.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()


class ModelWorker(QObject):
    """Runs one job at a time in its own thread. A job is a callable without arguments."""

    finished = Signal(str, object, str, float)

    @Slot(str, object)
    def run(self, name: str, job) -> None:
        started = time.perf_counter()
        try:
            result, error = job(), ""
        except mp.NothingToSubtract as exc:
            result, error = None, f"message:{exc}"
        except Exception:  # noqa: BLE001 - the error text goes to the UI unchanged
            result, error = None, traceback.format_exc()
        self.finished.emit(name, result, error, (time.perf_counter() - started) * 1000)


class Canvas(QWidget):
    """The photo with the mask overlay, the prompts, and the stroke and box input."""

    strokeFinished = Signal(list, bool)
    boxFinished = Signal(list, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(False)
        self.photo: QPixmap | None = None
        self.overlay: QImage | None = None
        self.overlay_visible = True
        self.tool = "stroke"
        self.enabled_input = False
        self.drag: list[QPointF] = []
        self.drag_subtract = False
        self.prompt_points: list[tuple[list[float], int]] = []
        self.prompt_boxes: list[list[float]] = []

    def set_photo(self, image: QImage | None) -> None:
        self.photo = QPixmap.fromImage(image) if image is not None else None
        self.overlay = None
        self.update()

    def photo_rect(self) -> QRectF:
        if self.photo is None:
            return QRectF()
        scale = min(self.width() / self.photo.width(), self.height() / self.photo.height())
        width, height = self.photo.width() * scale, self.photo.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def to_normalized(self, point: QPointF) -> list[float]:
        rect = self.photo_rect()
        x = (point.x() - rect.left()) / rect.width()
        y = (point.y() - rect.top()) / rect.height()
        return [min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)]

    def to_screen(self, point: list[float]) -> QPointF:
        rect = self.photo_rect()
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.photo is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a photo")
            return
        rect = self.photo_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(rect, self.photo, QRectF(self.photo.rect()))
        if self.overlay is not None and self.overlay_visible:
            painter.drawImage(rect, self.overlay, QRectF(self.overlay.rect()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for box in self.prompt_boxes:
            painter.setPen(QPen(QColor(255, 220, 0), 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self.to_screen(box[:2]), self.to_screen(box[2:])))
        for point, label in self.prompt_points:
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(QColor(0, 200, 80) if label == 1 else QColor(230, 40, 40))
            painter.drawEllipse(self.to_screen(point), 4, 4)
        if self.drag:
            color = QColor(230, 40, 40) if self.drag_subtract else QColor(0, 200, 80)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.tool == "box":
                painter.drawRect(QRectF(self.drag[0], self.drag[-1]).normalized())
            else:
                path = QPainterPath(self.drag[0])
                for point in self.drag[1:]:
                    path.lineTo(point)
                painter.drawPath(path)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (not self.enabled_input or self.photo is None
                or event.button() != Qt.MouseButton.LeftButton
                or not self.photo_rect().contains(event.position())):
            return
        self.drag = [event.position()]
        self.drag_subtract = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.drag:
            if self.tool == "box":
                self.drag = [self.drag[0], event.position()]
            else:
                self.drag.append(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if not self.drag or event.button() != Qt.MouseButton.LeftButton:
            return
        drag, self.drag = self.drag + [event.position()], []
        subtract = self.drag_subtract
        self.update()
        moved = max(abs(p.x() - drag[0].x()) + abs(p.y() - drag[0].y()) for p in drag)
        if self.tool == "box":
            if moved > CLICK_DRAG_LIMIT_SCREEN_PX:
                first, last = self.to_normalized(drag[0]), self.to_normalized(drag[-1])
                self.boxFinished.emit(first + last, subtract)
            return
        if moved <= CLICK_DRAG_LIMIT_SCREEN_PX:
            self.strokeFinished.emit([self.to_normalized(drag[0])], subtract)
        else:
            self.strokeFinished.emit([self.to_normalized(p) for p in drag], subtract)


class MainWindow(QMainWindow):
    submit = Signal(str, object)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle(f"AI Mask MVP ({args.provider})")
        self.pipeline: mp.MaskPipeline | None = None
        self.state: mp.PhotoState | None = None
        self.display_size = (0, 0)
        self.sky: mp.SkyResult | None = None
        self.people: mp.PeopleResult | None = None
        self.selection: mp.PromptSelection | None = None
        self.mask_kind: str | None = None
        self.people_ranks: list[int] | None = None
        self.busy = False
        self.event_count = 0
        self.overlay_color = QColor(255, 40, 80)

        self.worker = ModelWorker()
        self.worker_thread = QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.submit.connect(self.worker.run)
        self.worker.finished.connect(self.job_finished)
        self.worker_thread.start()

        self.canvas = Canvas()
        self.canvas.strokeFinished.connect(self.stroke_finished)
        self.canvas.boxFinished.connect(self.box_finished)
        self.overlay_timer = QTimer(self, singleShot=True, interval=OVERLAY_UPDATE_DELAY_MS)
        self.overlay_timer.timeout.connect(self.update_overlay)
        self.setCentralWidget(self.build_layout())
        self.build_actions()
        self.set_busy(True, "Loading models ...")
        self.submit.emit("load", self.load_job)

    # Layout.

    def build_layout(self) -> QWidget:
        panel = QVBoxLayout()
        open_button = QPushButton("Open photo ...")
        open_button.clicked.connect(self.choose_photo)
        panel.addWidget(open_button)

        one_click = QGroupBox("One click")
        grid = QGridLayout(one_click)
        self.model_buttons = []
        for index, (label, handler) in enumerate((("Subject", self.select_subject),
                                                  ("Background", self.select_background),
                                                  ("Sky", self.select_sky),
                                                  ("People", self.select_people))):
            button = QPushButton(label)
            button.clicked.connect(handler)
            grid.addWidget(button, index // 2, index % 2)
            self.model_buttons.append(button)
        panel.addWidget(one_click)

        self.people_list = QListWidget()
        self.people_list.setMaximumHeight(140)
        self.people_list.currentItemChanged.connect(self.person_chosen)
        panel.addWidget(QLabel("People list"))
        panel.addWidget(self.people_list)

        tools = QGroupBox("Selection (Alt subtracts)")
        tool_layout = QVBoxLayout(tools)
        self.stroke_tool = QRadioButton("Stroke and click")
        self.box_tool = QRadioButton("Box")
        self.stroke_tool.setChecked(True)
        group = QButtonGroup(self)
        for button in (self.stroke_tool, self.box_tool):
            group.addButton(button)
            tool_layout.addWidget(button)
            button.toggled.connect(self.tool_changed)
        row = QHBoxLayout()
        smaller, larger = QPushButton("Smaller [ "), QPushButton("Larger ]")
        smaller.clicked.connect(lambda: self.change_level(-1))
        larger.clicked.connect(lambda: self.change_level(1))
        row.addWidget(smaller)
        row.addWidget(larger)
        tool_layout.addLayout(row)
        new_selection = QPushButton("New selection")
        new_selection.clicked.connect(self.clear_selection)
        tool_layout.addWidget(new_selection)
        self.model_buttons += [smaller, larger, new_selection]
        panel.addWidget(tools)

        adjust = QGroupBox("Feather and Edge (photo px)")
        adjust_layout = QGridLayout(adjust)
        self.feather = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=FEATHER_RANGE_PX)
        self.edge = QSlider(Qt.Orientation.Horizontal, minimum=-EDGE_RANGE_PX,
                            maximum=EDGE_RANGE_PX)
        self.feather_label, self.edge_label = QLabel("0"), QLabel("0")
        for row_index, (name, slider, label) in enumerate((("Feather", self.feather,
                                                            self.feather_label),
                                                           ("Edge", self.edge, self.edge_label))):
            adjust_layout.addWidget(QLabel(name), row_index, 0)
            adjust_layout.addWidget(slider, row_index, 1)
            adjust_layout.addWidget(label, row_index, 2)
            slider.valueChanged.connect(self.adjustment_changed)
        panel.addWidget(adjust)

        overlay = QGroupBox("Overlay")
        overlay_layout = QGridLayout(overlay)
        self.overlay_visible = QCheckBox("Show (Y)", checked=True)
        self.overlay_visible.toggled.connect(self.overlay_toggled)
        self.prompts_visible = QCheckBox("Prompts", checked=True)
        self.prompts_visible.toggled.connect(self.refresh_prompts)
        color = QPushButton("Color ...")
        color.clicked.connect(self.choose_color)
        self.opacity = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=50)
        self.opacity.valueChanged.connect(lambda _: self.overlay_timer.start())
        overlay_layout.addWidget(self.overlay_visible, 0, 0)
        overlay_layout.addWidget(self.prompts_visible, 0, 1)
        overlay_layout.addWidget(color, 0, 2)
        overlay_layout.addWidget(QLabel("Opacity"), 1, 0)
        overlay_layout.addWidget(self.opacity, 1, 1, 1, 2)
        panel.addWidget(overlay)

        export = QPushButton("Export mask and prompts ...")
        export.clicked.connect(self.export)
        panel.addWidget(export)

        self.timing = QPlainTextEdit(readOnly=True)
        self.timing.setMaximumBlockCount(400)
        panel.addWidget(QLabel("Stage times (ms)"))
        panel.addWidget(self.timing, stretch=1)

        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(320)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(side)
        layout.addWidget(self.canvas, stretch=1)
        return root

    def build_actions(self) -> None:
        for text, key, handler in (("Open", QKeySequence.StandardKey.Open, self.choose_photo),
                                   ("Toggle overlay", QKeySequence("Y"),
                                    lambda: self.overlay_visible.toggle()),
                                   ("Smaller", QKeySequence("["), lambda: self.change_level(-1)),
                                   ("Larger", QKeySequence("]"), lambda: self.change_level(1))):
            action = QAction(text, self, shortcut=key)
            action.triggered.connect(handler)
            self.addAction(action)

    # Jobs.

    def load_job(self):
        paths = mp.GraphPaths.default(Path(self.args.explore))
        if self.args.subject_onnx:
            paths.subject_onnx = Path(self.args.subject_onnx)
        return mp.MaskPipeline(paths, provider=self.args.provider, threads=self.args.threads,
                               log_path=Path(self.args.log) if self.args.log else None)

    def run_job(self, name: str, job, message: str) -> None:
        if self.busy or self.pipeline is None:
            return
        self.set_busy(True, message)
        self.submit.emit(name, job)

    def set_busy(self, busy: bool, message: str = "") -> None:
        if busy and not self.busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        elif self.busy and not busy:
            QApplication.restoreOverrideCursor()
        self.busy = busy
        self.refresh_enabled()
        if message:
            self.statusBar().showMessage(message)

    def refresh_enabled(self) -> None:
        ready = not self.busy and self.state is not None
        self.canvas.enabled_input = ready
        for button in self.model_buttons:
            button.setEnabled(ready)

    @Slot(str, object, str, float)
    def job_finished(self, name: str, result, error: str, milliseconds: float) -> None:
        self.set_busy(False)
        if name == "load" and not error:
            self.pipeline = result
        self.show_new_events(name, milliseconds)
        if error.startswith("message:"):
            self.statusBar().showMessage(error[len("message:"):])
            return
        if error:
            QMessageBox.critical(self, f"{name} failed", error)
            if name == "load":
                self.close()
            self.statusBar().showMessage(f"{name} failed")
            return
        getattr(self, f"after_{name}")(result)

    def show_new_events(self, name: str, milliseconds: float) -> None:
        if self.pipeline is None:
            return
        events = self.pipeline.events[self.event_count:]
        self.event_count = len(self.pipeline.events)
        lines = []
        for event in events:
            detail = event.get("graph") or event.get("purpose") or event.get("photo", "")
            lines.append(f"{event['stage']:<10} {event['ms']:>9.1f}  {detail}")
        lines.append(f"-- {name}: {milliseconds:.1f} ms in total")
        self.timing.appendPlainText("\n".join(lines))

    def after_load(self, pipeline: mp.MaskPipeline) -> None:
        hashes = ", ".join(f"{k} {v['sha256'][:8]}" for k, v in pipeline.graph_records.items())
        self.statusBar().showMessage(f"Models loaded on {pipeline.provider}: {hashes}")
        if self.args.photo:
            self.open_photo(Path(self.args.photo))

    # Photo.

    def choose_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open photo", "",
                                              "Photos (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.open_photo(Path(path))

    def open_photo(self, path: Path) -> None:
        pipeline = self.pipeline

        def job():
            from PIL import Image

            state = pipeline.open_photo(path)
            display = state.photo.image.copy()
            display.thumbnail((DISPLAY_MAX_EDGE, DISPLAY_MAX_EDGE), Image.Resampling.BILINEAR)
            return state, np.asarray(display)

        self.run_job("open", job, f"Opening {path.name} ...")

    def after_open(self, result) -> None:
        self.state, display = result
        self.display_size = (display.shape[1], display.shape[0])
        self.sky = self.people = self.selection = None
        self.mask_kind = None
        self.people_list.clear()
        self.canvas.set_photo(rgb_qimage(display))
        self.refresh_enabled()
        photo = self.state.photo
        self.statusBar().showMessage(
            f"{photo.path.name}: {photo.width} x {photo.height} (file "
            f"{photo.original_size[0]} x {photo.original_size[1]})")
        self.refresh_prompts()

    # One-click selections.

    def select_subject(self) -> None:
        self.run_job("subject", lambda: self.pipeline.select_subject(self.state),
                     "Subject (BiRefNet) ...")

    def select_background(self) -> None:
        self.run_job("background", lambda: self.pipeline.select_subject(self.state),
                     "Background (BiRefNet) ...")

    def after_subject(self, _alpha) -> None:
        self.show_mask("subject", "Subject")

    def after_background(self, _alpha) -> None:
        self.show_mask("background", "Background")

    def select_sky(self) -> None:
        if self.sky is not None:
            self.after_sky(self.sky)
            return
        self.run_job("sky", lambda: self.pipeline.select_sky(self.state), "Sky ...")

    def after_sky(self, result: mp.SkyResult) -> None:
        self.sky = result
        if not result.found:
            self.statusBar().showMessage(result.message)
            QMessageBox.information(self, "Sky", result.message)
            return
        self.show_mask("sky", result.message)

    def select_people(self) -> None:
        if self.people is not None:
            self.after_people(self.people)
            return
        self.run_job("people", lambda: self.pipeline.select_people(self.state), "People ...")

    def after_people(self, result: mp.PeopleResult) -> None:
        self.people = result
        if not result.found:
            self.statusBar().showMessage(result.message)
            QMessageBox.information(self, "People", result.message)
            return
        self.people_list.blockSignals(True)
        self.people_list.clear()
        item = QListWidgetItem(f"All people ({len(result.people)})")
        item.setData(Qt.ItemDataRole.UserRole, None)
        self.people_list.addItem(item)
        for person in result.people:
            item = QListWidgetItem(f"Person {person['rank']} (score {person['score']:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, person["rank"])
            self.people_list.addItem(item)
        self.people_list.setCurrentRow(0)
        self.people_list.blockSignals(False)
        self.people_ranks = None
        self.show_mask("people", result.message)

    def person_chosen(self, item: QListWidgetItem | None, _previous) -> None:
        if item is None or self.people is None:
            return
        rank = item.data(Qt.ItemDataRole.UserRole)
        self.people_ranks = None if rank is None else [rank]
        self.show_mask("people", item.text())

    # Prompt selection.

    def tool_changed(self) -> None:
        self.canvas.tool = "box" if self.box_tool.isChecked() else "stroke"

    def active_selection(self) -> mp.PromptSelection:
        if self.selection is None or self.mask_kind != "selection":
            self.selection = self.pipeline.new_selection(self.state, self.args.stroke_points)
        return self.selection

    def stroke_finished(self, polyline: list, subtract: bool) -> None:
        if self.pipeline is None or self.state is None:
            return
        selection = self.active_selection()
        kind = "click" if len(polyline) == 1 else "stroke"
        self.run_job("selection", lambda: selection.add_stroke(polyline, subtract),
                     f"{'Subtract' if subtract else 'Add'} {kind} ...")

    def box_finished(self, box: list, subtract: bool) -> None:
        if self.pipeline is None or self.state is None:
            return
        if subtract:
            self.statusBar().showMessage("A subtract box is not available: the SAM 2.1 decoder "
                                         "takes one box as a positive prompt. Use a subtract "
                                         "stroke.")
            return
        selection = self.active_selection()
        self.run_job("selection", lambda: selection.set_box(box), "Box ...")

    def after_selection(self, _logits) -> None:
        step = self.selection.steps[-1]
        level = ""
        if self.selection.level_index is not None:
            level = (f", level {self.selection.level_index + 1} of {len(self.selection.levels)}"
                     f" ([ and ] change it)")
        self.show_mask("selection", f"Selection: {step['interaction']}, "
                                    f"{step['decoder_points']} decoder points{level}")

    def change_level(self, delta: int) -> None:
        if self.busy or self.selection is None or self.mask_kind != "selection":
            return
        if self.selection.change_level(delta):
            self.show_mask("selection", f"Level {self.selection.level_index + 1} of "
                                        f"{len(self.selection.levels)}")

    def clear_selection(self) -> None:
        self.selection = None
        if self.mask_kind == "selection":
            self.mask_kind = None
            self.canvas.overlay = None
            self.canvas.update()
        self.refresh_prompts()

    # Masks and overlay.

    def base_mask(self, width: int, height: int) -> np.ndarray | None:
        """The mask before Feather and Edge, in [0, 1], at width x height."""
        if self.mask_kind in ("subject", "background"):
            alpha = self.pipeline.subject_alpha_at_photo(self.state, width, height)
            return alpha if self.mask_kind == "subject" else 1.0 - alpha
        if self.mask_kind == "sky":
            return self.sky.mask(width, height).astype(np.float32)
        if self.mask_kind == "people":
            return self.people.mask(width, height, self.people_ranks).astype(np.float32)
        if self.mask_kind == "selection" and self.selection is not None:
            return mp.logits_to_mask(self.selection.logits, width, height).astype(np.float32)
        return None

    def show_mask(self, kind: str, message: str) -> None:
        self.mask_kind = kind
        self.statusBar().showMessage(message)
        self.refresh_prompts()
        self.update_overlay()

    def adjustment_changed(self) -> None:
        self.feather_label.setText(str(self.feather.value()))
        self.edge_label.setText(str(self.edge.value()))
        self.overlay_timer.start()

    def update_overlay(self) -> None:
        if self.state is None or self.mask_kind is None:
            return
        width, height = self.display_size
        mask = self.base_mask(width, height)
        scale = width / self.state.photo.width
        mask = mp.adjust_mask(mask, self.edge.value() * scale, self.feather.value() * scale)
        color = self.overlay_color
        rgba = np.empty((height, width, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color.red(), color.green(), color.blue()
        rgba[..., 3] = np.round(mask * (self.opacity.value() / 100.0) * 255).astype(np.uint8)
        self.canvas.overlay = rgba_qimage(rgba)
        self.canvas.update()

    def overlay_toggled(self, visible: bool) -> None:
        self.canvas.overlay_visible = visible
        self.canvas.update()

    def choose_color(self) -> None:
        color = QColorDialog.getColor(self.overlay_color, self, "Overlay color")
        if color.isValid():
            self.overlay_color = color
            self.update_overlay()

    def refresh_prompts(self) -> None:
        points, boxes = [], []
        if self.prompts_visible.isChecked():
            if self.mask_kind == "selection" and self.selection is not None:
                points = list(zip(self.selection.points, self.selection.labels))
                boxes = [self.selection.box] if self.selection.box else []
            elif self.mask_kind == "sky" and self.sky is not None:
                points = [([x / 1024, y / 1024], 1) for region in self.sky.regions
                          for x, y in region["points"]]
            elif self.mask_kind == "people" and self.people is not None:
                boxes = [[v / 1024 for v in person["box_model"]] for person in self.people.people
                         if self.people_ranks is None or person["rank"] in self.people_ranks]
        self.canvas.prompt_points = points
        self.canvas.prompt_boxes = boxes
        self.canvas.update()

    # Export.

    def export(self) -> None:
        if self.state is None or self.mask_kind is None:
            self.statusBar().showMessage("No mask to export.")
            return
        default = str(self.state.photo.path.with_name(
            f"{self.state.photo.path.stem}_{self.mask_kind}_mask.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Export mask", default, "PNG (*.png)")
        if not path:
            return
        photo = self.state.photo
        mask = mp.adjust_mask(self.base_mask(photo.width, photo.height), self.edge.value(),
                              self.feather.value())
        mp.save_mask_png(mask, Path(path))
        prompts = {"photo": str(photo.path), "size": [photo.width, photo.height],
                   "original_size": list(photo.original_size), "mask": self.mask_kind,
                   "feather_px": self.feather.value(), "edge_px": self.edge.value()}
        if self.mask_kind == "sky":
            prompts["sky"] = mp.to_jsonable(self.sky.__dict__)
        elif self.mask_kind == "people":
            prompts["people"] = mp.to_jsonable(self.people.__dict__)
            prompts["people_ranks"] = self.people_ranks
        elif self.mask_kind == "selection":
            prompts["selection"] = self.selection.record()
        prompts["pipeline"] = {k: v for k, v in self.pipeline.log().items() if k != "events"}
        mp.save_json(Path(path).with_suffix(".json"), prompts)
        self.statusBar().showMessage(f"Exported {path} and its prompts JSON")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", nargs="?", default="")
    parser.add_argument("--provider", choices=sorted(mp.PROVIDERS), default="directml")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--explore", default=str(mp.DEFAULT_EXPLORE),
                        help="build/tmp/explore of this repository (the graph locations of "
                             "mask_pipeline.GraphPaths.default)")
    parser.add_argument("--subject-onnx", default="",
                        help="another BiRefNet lite 1024 graph (default: the E2 float16 graph)")
    parser.add_argument("--stroke-points", type=int, default=mp.STROKE_POINT_COUNT)
    parser.add_argument("--log", default="", help="JSON record of the stage times and hashes")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1500, 950)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
