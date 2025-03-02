# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# List of hidden imports
hidden_imports = [
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'numpy',
    'torch',
    'tqdm',
    'transformers',
    'speech_recognition',
]

# Collecting data files
datas = [
    # If you have additional data files like models, images, etc., add them here
    # Example: ('path/to/datafile', 'destination_directory'),
]

# Analysis
a = Analysis(
    ['AI3v13.py'],
    pathex=[],  # Add paths to any other directories needed for your script
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

# Create a PYZ archive with collected Python files
pyz = PYZ(a.pure, cipher=block_cipher)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='AI3v13',
    debug=False,  # Set to True to enable debug mode
    strip=False,
    upx=True,  # Set to False if you encounter issues with UPX
    console=True,  # Set to False for windowed apps
    cipher=block_cipher
)

# Create a one-folder bundle containing the above executable
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='AI3v13',
)
