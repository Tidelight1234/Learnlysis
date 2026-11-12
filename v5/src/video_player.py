# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""视频播放模块 VideoPlayerModule。"""

import os
import sys

from PySide6.QtWidgets import QWidget, QPushButton, QLineEdit, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QListWidget, QListView, QMessageBox, QLayout
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, QTimer, Signal
from PySide6.QtGui import QIntValidator, QColor, QPalette

import theme
import user_state

# 让 Qt 多媒体优先使用 Windows Media Foundation 后端（MFPlayerService）：它调用系统原生解码器，
# MP4/H.264、MOV、MKV 等开箱即播；默认的 DirectShow 后端在缺解码器时无法播放 MP4。
# 必须在创建 QMediaPlayer 之前设置：本模块一被导入就设置，而 QMediaPlayer 只在 __init__ 中创建，
# 因此无论程序从哪个入口启动都能生效；若插件缺失 Qt 会自动回退，不影响 AVI 等既有格式。
os.environ.setdefault("QT_MULTIMEDIA_PREFERRED_PLUGINS", "windowsmediafoundation")

if sys.platform == "win32":
    import ctypes


def _refresh_native_bg(widget):
    """让原生窗体立即按当前 palette 重画整块客户区。

    QVideoWidget 有自己的 HWND：Qt 的 repaint()/update() 只会把缓存的旧位图贴回屏幕，
    不会让 Win32 客户区失效，所以主题切换后即使 palette 已改，屏幕上仍是旧底色。
    InvalidateRect(bErase=TRUE) 触发 WM_ERASEBKGND，Qt 在里面用当前 palette 重新填底，
    切换瞬间即同步；非 Windows 平台没有这层缓存，走常规 update() 即可。
    """
    if sys.platform != "win32":
        widget.update()
        return
    try:
        hwnd = ctypes.c_void_p(int(widget.winId()))
        user32 = ctypes.windll.user32
        user32.InvalidateRect(hwnd, None, True)
        user32.UpdateWindow(hwnd)
    except Exception:
        # 刷新失败只是旧色多留一帧，不能拖垮主题切换
        widget.update()


SEEK_STEP = 5  # 快进/快退步长（秒）
SPEEDS = (("0.25×", 0.25), ("0.5×", 0.5), ("0.75×", 0.75), ("1.0×", 1.0),
          ("1.25×", 1.25), ("1.5×", 1.5), ("2.0×", 2.0))
# 打开视频对话框的过滤器：默认展示所有常见视频容器，兼容 MP4/AVI/MKV/MOV 等。
# 能否真正播放取决于系统解码器与多媒体后端（见 main.py 的 WMF 设置）。
VIDEO_FILTER = (
    "所有视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.mpg *.mpeg *.m4v *.3gp *.ts *.rmvb);;"
    "MP4 (*.mp4);;AVI (*.avi);;MKV (*.mkv);;MOV (*.mov);;WMV (*.wmv);;"
    "FLV (*.flv);;WebM (*.webm);;所有文件 (*)"
)

class VideoPlayerModule(QWidget):
    # 片段起点变更信号：编码区“当前时间点”等外部展示跟随此值刷新
    segment_start_changed = Signal(int)
    _current_position = 0

    @property
    def current_position(self):
        """当前播放片段的起点（秒）。"""
        return self._current_position

    @current_position.setter
    def current_position(self, seconds):
        # 写成属性：所有既有赋值语句自动广播变更，无需逐处补发信号
        self._current_position = seconds
        self.segment_start_changed.emit(seconds)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_position = 0
        self.segment_duration = 10
        self.current_speed = 1.0
        self.current_file_path = ""
    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)
        # 整块面板是一张卡片：视频舞台 → 播放条 → 参数行 → 播放列表
        panel = QWidget()
        panel.setProperty("card", True)
        self.panel = panel
        # 不给面板加 QGraphicsEffect：它会把子树重绘进位图，
        # 原生 QVideoWidget 进不了这张位图，还会随每帧刷新拖慢整块面板；
        # 投影改在 paintEvent 里手绘（见 paintEvent）
        panel_layout = QVBoxLayout(panel)
        self.panel_layout = panel_layout
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)
        layout.addWidget(panel)
        # 视频显示区域
        self.video_widget = QVideoWidget()
        # 占位底色走 palette（见 _apply_placeholder_bg）；开 autofill 兜底，
        # 保证全屏/原生窗体路径下背景依旧被填上
        self.video_widget.setAutoFillBackground(True)
        self._apply_placeholder_bg()
        video_stage = QWidget()
        self.video_stage = video_stage
        video_stage.setProperty("videoStage", True)
        video_stage.setMinimumHeight(200)
        stage_layout = QVBoxLayout(video_stage)
        stage_layout.setContentsMargins(6, 6, 6, 6)
        stage_layout.setSpacing(0)
        stage_layout.addWidget(self.video_widget)
        panel_layout.addWidget(video_stage, 1)
        # 不要传 QMediaPlayer.VideoSurface：该 flag 要求后端提供 QVideoSurfaceControl，
        # 而 WMF 后端只提供 QVideoWindowControl，会导致 Qt 跳过 WMF、回退到 DirectShow，
        # 进而使 MP4 因缺解码器而无法播放（实测：加 flag=DirectShow 放不了，不加 flag=MFPlayerService 正常播）。
        # QVideoWidget 走窗口输出即可，无需该 flag。
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        # 播放控制按钮（纯图标，语义靠 tooltip 承载）
        self.replay_button = QPushButton()
        self.replay_button.setProperty("btn", "round")
        self.replay_button.setFixedSize(30, 30)
        self.replay_button.setToolTip("重播上一片段")
        theme.bind_icon(self.replay_button, "replay", 15)
        self.rewind_button = QPushButton()
        self.rewind_button.setProperty("btn", "round")
        self.rewind_button.setFixedSize(30, 30)
        self.rewind_button.setToolTip(f"快退 {SEEK_STEP} 秒")
        theme.bind_icon(self.rewind_button, "skip-back", 15)
        self.play_button = QPushButton()
        self.play_button.setProperty("btn", "play")
        self.play_button.setFixedSize(36, 36)
        self.play_button.setToolTip("播放")
        theme.bind_icon(self.play_button, "play", 16, "#FFFFFF")
        self.forward_button = QPushButton()
        self.forward_button.setProperty("btn", "round")
        self.forward_button.setFixedSize(30, 30)
        self.forward_button.setToolTip(f"快进 {SEEK_STEP} 秒")
        theme.bind_icon(self.forward_button, "skip-forward", 15)
        self.jump_button = QPushButton()
        self.jump_button.setProperty("btn", "round")
        self.jump_button.setFixedSize(30, 30)
        self.jump_button.setToolTip("跳转到指定时间")
        theme.bind_icon(self.jump_button, "jump", 15)
        # 进度控制区域
        self.progress_slider = QSlider(Qt.Horizontal)
        if sys.platform == "darwin":
            self.progress_slider.setProperty("macOptimized", True)
            self.progress_slider.setMinimumHeight(24)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setToolTip("播放进度")
        self.current_time_label = QLabel("00:00 / 00:00")
        self.current_time_label.setProperty("timecode", True)
        self.current_time_label.setAlignment(Qt.AlignCenter)
        # 预留过一小时时长“1:23:45 / 1:23:45”的宽度，否则文本会被裁掉
        self.current_time_label.setFixedWidth(132)
        # 播放速度
        self.speed_combo = QComboBox()
        self.speed_combo.setProperty("pill", True)
        self.speed_combo.setFixedSize(78, 30)
        self.speed_combo.setToolTip("播放速度")
        speed_view = QListView()
        speed_view.setObjectName("speedPopup")
        speed_view.setSpacing(2)
        speed_view.setMinimumWidth(112)
        self.speed_combo.setView(speed_view)
        # 容器可能被 Qt 重建，改用绑定器持续施加圆角弹层配方（含 mask 兜底）
        theme.bind_combo_popup(self.speed_combo)
        transport_layout = QHBoxLayout()
        transport_layout.setContentsMargins(2, 0, 2, 0)
        transport_layout.setSpacing(8)
        transport_layout.addWidget(self.replay_button)
        transport_layout.addWidget(self.rewind_button)
        transport_layout.addWidget(self.play_button)
        transport_layout.addWidget(self.forward_button)
        transport_layout.addSpacing(8)
        transport_layout.addWidget(self.current_time_label)
        transport_layout.addWidget(self.progress_slider, 1)
        transport_layout.addWidget(self.speed_combo)
        panel_layout.addLayout(transport_layout)
        # 片段时长与时间点跳转
        self.jump_inpouthour = QLineEdit()
        self.jump_inpouthour.setPlaceholderText("时")
        self.jump_inpouthour.setAlignment(Qt.AlignCenter)
        self.jump_inpouthour.setFixedSize(44, 30)
        self.jump_inpouthour.setToolTip("跳转到的小时（留空视为 0）")
        self.jump_inpouthour.setValidator(QIntValidator(0, 23, self))
        self.jump_inputlabelhour = QLabel(':')
        self.jump_inputlabelhour.setAlignment(Qt.AlignCenter)
        self.jump_inputlabelhour.setFixedWidth(8)
        self.jump_inputmin = QLineEdit()
        self.jump_inputmin.setPlaceholderText("分")
        self.jump_inputmin.setAlignment(Qt.AlignCenter)
        self.jump_inputmin.setFixedSize(44, 30)
        self.jump_inputmin.setToolTip("跳转到的分钟")
        self.jump_inputlabel = QLabel(':')
        self.jump_inputlabel.setAlignment(Qt.AlignCenter)
        self.jump_inputlabel.setFixedWidth(8)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("秒")
        self.jump_input.setAlignment(Qt.AlignCenter)
        self.jump_input.setFixedSize(44, 30)
        self.jump_input.setToolTip("跳转到的秒")
        # 添加整数验证器，限制输入范围为 0~59
        self.jump_inputmin.setValidator(QIntValidator(0, 59, self))
        self.jump_input.setValidator(QIntValidator(0, 59, self))
        self.duration_slider = QSlider(Qt.Horizontal)
        if sys.platform == "darwin":
            self.duration_slider.setProperty("macOptimized", True)
            self.duration_slider.setMinimumHeight(24)
        self.duration_slider.setRange(1, 30)  # 默认最大为30秒
        self.duration_slider.setValue(10)     # 默认播放10秒
        self.duration_slider.setMinimumWidth(120)
        self.duration_slider.setToolTip("每个片段自动播放的时长")
        self.time_label = QLabel(f"片段时长 {self.duration_slider.value()} 秒")
        self.time_label.setProperty("hint", True)
        param_layout = QHBoxLayout()
        param_layout.setContentsMargins(2, 0, 2, 0)
        param_layout.setSpacing(8)
        param_layout.addWidget(self.time_label)
        param_layout.addWidget(self.duration_slider, 1)
        param_layout.addSpacing(12)
        param_layout.addWidget(self.jump_inpouthour)
        param_layout.addWidget(self.jump_inputlabelhour)
        param_layout.addWidget(self.jump_inputmin)
        param_layout.addWidget(self.jump_inputlabel)
        param_layout.addWidget(self.jump_input)
        param_layout.addWidget(self.jump_button)
        panel_layout.addLayout(param_layout)
        # 播放列表
        self.file_label = QLabel("视频列表")
        self.file_label.setProperty("section", True)
        self.open_button = QPushButton()
        self.open_button.setProperty("btn", "round")
        self.open_button.setFixedSize(30, 30)
        self.open_button.setToolTip("打开视频文件")
        theme.bind_icon(self.open_button, "folder", 15)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(132)
        self.file_list.setSpacing(2)
        self.file_list.setTextElideMode(Qt.ElideRight)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.file_list.setToolTip("双击播放")
        playlist_header = QHBoxLayout()
        playlist_header.setContentsMargins(2, 0, 2, 0)
        playlist_header.setSpacing(8)
        playlist_header.addWidget(self.file_label)
        playlist_header.addStretch()
        playlist_header.addWidget(self.open_button)
        panel_layout.addLayout(playlist_header)
        panel_layout.addWidget(self.file_list)
        # 初始化组件
        self._init_components()

    def set_compact_layout(self, compact):
        """低高度时压缩可伸缩区域，保留全部核心控件。"""
        compact = bool(compact)
        if getattr(self, "_compact_layout", None) == compact:
            return
        self._compact_layout = compact
        self.video_stage.setMinimumHeight(140 if compact else 200)
        self.file_list.setMaximumHeight(76 if compact else 132)
        margins = (8, 8, 8, 8) if compact else (12, 12, 12, 12)
        self.panel_layout.setContentsMargins(*margins)
        self.panel_layout.setSpacing(6 if compact else 10)
        self.updateGeometry()
    def _init_components(self):
        # 初始化播放器组件
        for label, rate in SPEEDS:
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(3)
        # 信号连接
        self.open_button.clicked.connect(self.open_file)
        self.file_list.itemDoubleClicked.connect(self.load_video_from_list)
        # 恢复存档时 main_window 会直接往列表里 addItem，规范显示要走同一个入口
        self.file_list.model().rowsInserted.connect(self.normalize_playlist)
        self.play_button.clicked.connect(self.play_segment)
        self.replay_button.clicked.connect(self.replay_segment)
        self.rewind_button.clicked.connect(lambda: self.seek_by(-SEEK_STEP))
        self.forward_button.clicked.connect(lambda: self.seek_by(SEEK_STEP))
        self.jump_button.clicked.connect(self.jump_to_time)
        self.duration_slider.valueChanged.connect(self.update_duration)
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        self.progress_slider.sliderMoved.connect(self.set_position)
        self.media_player.positionChanged.connect(self.update_progress_bar)
        self.media_player.durationChanged.connect(self.update_duration_bar)
        self.media_player.playbackStateChanged.connect(self.update_play_icon)
        self.media_player.errorOccurred.connect(self.handle_media_error)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        # 主题切换后原生 QVideoWidget 不会立即重画，订阅信号同步更新占位背景
        theme.themeChanged.connect(self._on_theme_changed)
    def paintEvent(self, event):
        """给视频卡片补上与其他模块一致的投影（含原生窗体，不能用图形效果）。"""
        super().paintEvent(event)
        theme.paint_card_shadow(self, getattr(self, "panel", None))
    def _apply_placeholder_bg(self):
        """把当前主题的占位色直写 QVideoWidget 的 palette。

        无视频时 QVideoWidget::paintEvent 就是用 palette().color(QPalette::Window) 填底，
        所以颜色跟着 token 走；不走样式表是因为 QSS 对原生窗体要等延迟的 style polish。
        只改 palette 屏幕不会立即变（见 _refresh_native_bg）。
        """
        pal = self.video_widget.palette()
        pal.setColor(QPalette.Window, QColor(theme.tokens()["video_bg"]))
        self.video_widget.setPalette(pal)
    def _on_theme_changed(self, _mode):
        """主题切换后立即重绘视频占位背景，避免原生窗体残留旧色。"""
        self._apply_placeholder_bg()
        self.update()
        if not self.video_widget.isVisible():
            # 还没显示就不用强刷：下次显示时本来就会用新 palette 首次绘制
            return
        if self.media_player.mediaStatus() in (QMediaPlayer.MediaStatus.NoMedia,
                                               QMediaPlayer.MediaStatus.InvalidMedia):
            # 无画面：屏上看到的就是占位底色，失效整块原生客户区立即换上新色
            _refresh_native_bg(self.video_widget)
        else:
            # 有画面：视频帧盖住底色，失效重画反而会让占位色在画面上闪一下
            self.video_widget.update()
    def play_segment(self):
        if not self.media_player.isAvailable():
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.setPosition(self.current_position * 1000)
            self.media_player.play()
            self.timer.start(100)  # 启动定时器每100毫秒更新一次
    def _on_media_status_changed(self, status):
        """Qt 6 异步设置 source；媒体状态变化后更新视频占位层。"""
        if status in (QMediaPlayer.MediaStatus.NoMedia,
                      QMediaPlayer.MediaStatus.InvalidMedia):
            _refresh_native_bg(self.video_widget)
    def replay_segment(self):
        if not self.media_player.isAvailable():
            return
        self.current_position -= self.segment_duration
        if self.current_position < 0:
            self.current_position = 0
        self.media_player.setPosition(self.current_position * 1000)
        self.media_player.play()
        self.timer.start(100)  # 启动定时器每100毫秒更新一次
    def seek_by(self, delta_seconds):
        """快进/快退：相对当前位置微调，并把片段起点同步过去。"""
        duration = self.media_player.duration()
        if duration <= 0:
            return
        target = self.media_player.position() + delta_seconds * 1000
        target = max(0, min(target, duration))
        self.media_player.setPosition(target)
        self.current_position = target // 1000
    def update_play_icon(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        theme.set_icon(self.play_button, "pause" if playing else "play", 16, "#FFFFFF")
        self.play_button.setToolTip("暂停" if playing else "播放")
    def handle_media_error(self, _error):
        """媒体加载或解码失败时给出可读提示。

        常见于系统缺少该容器/编码对应的解码器（如某些 MKV、HEVC 视频）。
        """
        detail = self.media_player.errorString()
        QMessageBox.warning(
            self, "无法播放",
            "该视频无法播放，可能是格式不受支持或缺少相应解码器。\n\n"
            f"文件：{self.current_file_path or '（未加载）'}\n"
            f"详情：{detail or '未知错误'}"
        )
    def jump_to_time(self):
        try:
            # 获取用户输入的时、分和秒（时留空视为 0）
            hours_text = self.jump_inpouthour.text().strip()
            minutes_text = self.jump_inputmin.text()
            seconds_text = self.jump_input.text()
            if not minutes_text or not seconds_text:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("无效输入")
                msg_box.setText("请输入有效的分与秒！")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.exec()
                return
            hours = int(hours_text) if hours_text else 0
            minutes = int(minutes_text)
            seconds = int(seconds_text)
            if not 0 <= hours <= 23 or not 0 <= minutes <= 59 or not 0 <= seconds <= 59:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("无效输入")
                msg_box.setText("时须在 0~23，分与秒须在 0~59 之间！")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.exec()
                return
            # 计算总秒数
            time_point = hours * 3600 + minutes * 60 + seconds
            # 获取视频总时长（秒）
            total_duration = self.media_player.duration() // 1000
            # 判断是否超过总时长
            if time_point > total_duration:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("无效时间")
                msg_box.setText("跳转时间超过视频总时长")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.exec()
                return
            # 设置当前播放位置
            self.current_position = time_point
            self.media_player.setPosition(time_point * 1000)
        except ValueError:
            QMessageBox.warning(self, "无效输入", "请输入有效的数字！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"跳转失败: {str(e)}")
    def update_duration(self, value):
        self.segment_duration = value
        self.time_label.setText(f"片段时长 {value} 秒")
    def change_speed(self, index):
        self.current_speed = float(self.speed_combo.itemData(index))
        self.media_player.setPlaybackRate(self.current_speed)
    def open_file(self):
        files = user_state.open_file_names(self, user_state.BUCKET_VIDEO,
                                          "打开视频文件", VIDEO_FILTER)
        if files:
            self.add_to_playlist(files)
    def add_to_playlist(self, files):
        """把一批视频加进播放列表并高亮第一个（不自动播放，沿用双击列表才开播的老口径）。"""
        for path in files:
            if self._find_playlist_item(path) is None:
                self.file_list.addItem(path)
        self.normalize_playlist()
        self.current_file_path = files[0]  # 记录第一个文件路径
        self.highlight_current_file()

    def _find_playlist_item(self, file_path):
        """按完整路径找列表项：列表里只显文件名，去重不能靠文本比。"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if (item.data(Qt.UserRole) or item.text()) == file_path:
                return item
        return None

    def reveal_in_playlist(self, file_path):
        """从「最近打开」重开一个视频：不在列表里就补上，然后按老口径装载开播。"""
        if self._find_playlist_item(file_path) is None:
            self.file_list.addItem(file_path)
            self.normalize_playlist()
        self.load_video(file_path)
        return file_path
    def normalize_playlist(self, *_args):
        """列表项统一显示文件名，完整路径存进 UserRole。"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            path = item.data(Qt.UserRole)
            if not path:
                path = item.text()
                item.setData(Qt.UserRole, path)
                item.setText(path.replace("\\", "/").rsplit("/", 1)[-1])
                item.setToolTip(path)
        self.highlight_current_file()
    def highlight_current_file(self):
        """高亮当前视频对应的列表项，替代原先的“当前文件”文字标签。"""
        if not self.current_file_path:
            return
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == self.current_file_path:
                self.file_list.setCurrentItem(item)
                return
    def load_video_from_list(self, item):
        file_path = item.data(Qt.UserRole) or item.text()
        self.current_file_path = file_path     # 更新当前文件路径
        self.load_video(file_path)
    def load_video(self, file_path):
        if not file_path or not os.path.exists(file_path):
            # 路径不存在时不交给播放器，避免对失效路径触发无意义的解码错误弹窗
            return
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.current_position = 0
        self.media_player.setPosition(0)
        self.media_player.play()
        self.media_player.pause()  # 立即暂停以显示第一帧
        self.timer.start(100)  # 启动定时器每100毫秒更新一次
        if file_path != self.current_file_path:
            self.current_file_path = file_path
        self.highlight_current_file()
    def set_position(self, position):
        self.media_player.setPosition(position)
        self.current_position = position // 1000
    def update_progress(self):
        position = self.media_player.position()
        current_time = self.format_time(position)
        total_time = self.format_time(self.media_player.duration())
        self.current_time_label.setText(f"{current_time} / {total_time}")
        if position >= (self.current_position + self.segment_duration) * 1000:
            self.media_player.pause()
            self.current_position += self.segment_duration
    def update_progress_bar(self, position):
        self.progress_slider.setValue(position)
    def update_duration_bar(self, duration):
        self.progress_slider.setRange(0, duration)
    def format_time(self, milliseconds):
        seconds = int(milliseconds / 1000)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # 满一小时才补时位：短视频保持“05:12 / 58:40”的紧凑写法
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
