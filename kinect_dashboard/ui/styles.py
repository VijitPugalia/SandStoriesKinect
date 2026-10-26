"""Design system and stylesheets for SandStories Azure Kinect Capture Dashboard.

Compliant with Web Interface Guidelines:
- High-contrast Obsidian / Zinc dark theme (#09090B / #18181B / #27272A).
- Accessible 2px focus rings on all interactive elements (:focus / :focus-visible).
- Monospace tabular numeric telemetry ("Cascadia Code", monospace) to prevent jitter.
- Tactile button states (hover, pressed, disabled) and generous hit targets (>= 32px).
- Zero heavy drop shadows; clean 1px hairline structural borders.
"""

DARK_THEME_STYLESHEET = """
/* ── Base & Window ─────────────────────────────────────────────────── */
QMainWindow, QDialog, QWidget#CentralWidget {
    background-color: #09090B;
    color: #F4F4F5;
    font-family: "Segoe UI Variable Text", "Segoe UI", -apple-system, sans-serif;
    font-size: 13px;
}

QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 1px;
}
QScrollBar::handle:vertical {
    background: #27272A;
    min-height: 28px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #3F3F46;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── App Header ────────────────────────────────────────────────────── */
QWidget#AppHeader {
    background-color: #0C0C0E;
    border-bottom: 1px solid #27272A;
}

QLabel#BrandTitle {
    font-size: 15px;
    font-weight: 700;
    color: #FAFAFA;
    letter-spacing: -0.2px;
}

QLabel#BrandSep {
    font-size: 13px;
    color: #52525B;
    padding: 0 4px;
}

QLabel#BrandSubtitle {
    font-size: 12px;
    font-weight: 500;
    color: #A1A1AA;
    letter-spacing: 0.1px;
}

QLabel#MasterTimer {
    font-family: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
    font-size: 20px;
    font-weight: 600;
    color: #FAFAFA;
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 6px;
    padding: 3px 14px;
    letter-spacing: 1px;
    qproperty-alignment: 'AlignCenter';
}

QLabel#StatusBadge {
    font-family: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
    font-weight: 700;
    font-size: 11px;
    border-radius: 14px;
    padding: 5px 14px;
    letter-spacing: 0.8px;
}

QLabel#SensorPill {
    font-size: 11px;
    font-weight: 600;
    color: #A1A1AA;
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 14px;
    padding: 5px 13px;
    letter-spacing: 0.3px;
}

/* ── Bottom Control Bar ────────────────────────────────────────────── */
QWidget#BottomBar {
    background-color: #0C0C0E;
    border-top: 1px solid #27272A;
}

QLabel#GestureHint {
    font-family: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #71717A;
    letter-spacing: 0.2px;
}

/* ── Sidebar Panels ────────────────────────────────────────────────── */
QScrollArea#SetupRail {
    background-color: #0C0C0E;
    border-left: 1px solid #27272A;
}

QWidget#SetupRailContent {
    background-color: transparent;
}

QLabel#RailTitle {
    font-size: 12px;
    font-weight: 700;
    color: #A1A1AA;
    letter-spacing: 1.2px;
    padding: 4px 6px 10px 6px;
}

QLabel#SectionTitle {
    font-size: 10px;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 1.2px;
    padding: 14px 6px 6px 6px;
    border-top: 1px solid #27272A;
}

QLabel#SectionHint {
    font-size: 11px;
    color: #71717A;
    line-height: 1.4;
    padding: 2px 2px 8px 2px;
}

QLabel#FieldLabel {
    font-size: 11px;
    font-weight: 600;
    color: #A1A1AA;
    letter-spacing: 0.1px;
}

QLabel#FieldValue {
    font-size: 12px;
    color: #F4F4F5;
    font-weight: 500;
}

QLabel#MonoReadout {
    font-family: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #38BDF8;
    font-weight: 600;
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 3px 7px;
}

QLabel#PreviewPath, QLabel#PathReadout {
    font-family: "Cascadia Code", "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
    color: #CBD2D8;
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 6px 8px;
}

/* ── Form Controls (Inputs, Selectors, Spins) ──────────────────────── */
QComboBox, QLineEdit, QSpinBox {
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 6px;
    padding: 5px 9px;
    color: #FAFAFA;
    font-size: 12px;
    min-height: 22px;
    selection-background-color: #2563EB;
}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover {
    border-color: #3F3F46;
    background-color: #151518;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus {
    border: 2px solid #3B82F6;
    background-color: #18181B;
    outline: none;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #71717A;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #121215;
    border: 1px solid #27272A;
    selection-background-color: #2563EB;
    color: #FAFAFA;
    padding: 4px;
    border-radius: 6px;
    outline: none;
}

QCheckBox {
    spacing: 10px;
    color: #A1A1AA;
    font-size: 11px;
    min-height: 24px;
}
QCheckBox:hover {
    color: #FAFAFA;
}
QCheckBox:focus {
    color: #FAFAFA;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3F3F46;
    background: #121215;
}
QCheckBox::indicator:hover {
    border-color: #3B82F6;
}
QCheckBox::indicator:checked {
    background: #16A34A;
    border-color: #22C55E;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #18181B;
    border: 1px solid #27272A;
    border-radius: 6px;
    color: #FAFAFA;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 15px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #27272A;
    border-color: #3F3F46;
}
QPushButton:pressed {
    background-color: #121215;
}
QPushButton:focus {
    border: 2px solid #3B82F6;
    outline: none;
}
QPushButton:disabled {
    background-color: #0F0F12;
    border-color: #1F1F23;
    color: #52525B;
}

QPushButton#BtnGhost {
    background-color: transparent;
    border: 1px solid #27272A;
    color: #A1A1AA;
    font-size: 11px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 5px;
}
QPushButton#BtnGhost:hover {
    color: #FAFAFA;
    border-color: #3F3F46;
    background-color: #18181B;
}
QPushButton#BtnGhost:focus {
    border: 2px solid #3B82F6;
    outline: none;
}

QPushButton#BtnSidebarToggle {
    background-color: #121215;
    border: 1px solid #27272A;
    border-radius: 6px;
    color: #A1A1AA;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 12px;
    letter-spacing: 0.2px;
}
QPushButton#BtnSidebarToggle:hover {
    color: #FAFAFA;
    background-color: #18181B;
    border-color: #3F3F46;
}
QPushButton#BtnSidebarToggle:focus {
    border: 2px solid #3B82F6;
    outline: none;
}

QPushButton#BtnStart {
    background-color: #DC2626;
    border: 1px solid #EF4444;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 700;
    min-height: 42px;
    min-width: 154px;
    border-radius: 21px;
    padding: 8px 24px;
    letter-spacing: 0.3px;
}
QPushButton#BtnStart:hover {
    background-color: #EF4444;
    border-color: #F87171;
}
QPushButton#BtnStart:pressed {
    background-color: #B91C1C;
}
QPushButton#BtnStart:focus {
    border: 2px solid #60A5FA;
    outline: none;
}
QPushButton#BtnStart:disabled {
    background-color: #18181B;
    border-color: #27272A;
    color: #52525B;
}

QPushButton#BtnPause {
    background-color: #18181B;
    border: 1px solid #3F3F46;
    color: #F4F4F5;
    min-width: 96px;
    min-height: 38px;
    border-radius: 19px;
    font-weight: 600;
}
QPushButton#BtnPause:hover {
    background-color: #27272A;
    border-color: #F59E0B;
    color: #FBBF24;
}
QPushButton#BtnPause:focus {
    border: 2px solid #3B82F6;
    outline: none;
}

QPushButton#BtnStop {
    background-color: #18181B;
    border: 1px solid #3F3F46;
    color: #F87171;
    min-width: 116px;
    min-height: 38px;
    border-radius: 19px;
    font-weight: 600;
}
QPushButton#BtnStop:hover {
    background-color: #2B1113;
    border-color: #EF4444;
    color: #FFFFFF;
}
QPushButton#BtnStop:focus {
    border: 2px solid #3B82F6;
    outline: none;
}
QPushButton#BtnStop:disabled {
    background-color: #121215;
    border-color: #27272A;
    color: #52525B;
}

QPushButton#BtnDiscard {
    background-color: transparent;
    border: 1px solid transparent;
    color: #71717A;
    font-size: 11px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 6px;
}
QPushButton#BtnDiscard:hover {
    color: #EF4444;
    border-color: #7F1D1D;
    background-color: #1C1012;
}
QPushButton#BtnDiscard:focus {
    border: 2px solid #3B82F6;
    outline: none;
}
QPushButton#BtnDiscard:disabled {
    color: #27272A;
    border-color: transparent;
    background-color: transparent;
}

QSplitter::handle {
    background-color: #27272A;
    width: 1px;
}
QSplitter::handle:hover {
    background-color: #3B82F6;
}

QStatusBar {
    background-color: #09090B;
    border-top: 1px solid #27272A;
    color: #71717A;
    font-size: 11px;
    padding: 4px 14px;
}
"""
