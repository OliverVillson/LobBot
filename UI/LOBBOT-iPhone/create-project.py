"""Create a portable, shared-scheme Xcode project and local brand assets."""
from pathlib import Path
import hashlib
import json
import plistlib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'LOBBOT'
PROJECT = ROOT / 'LOBBOT.xcodeproj'
PROJECT.mkdir(exist_ok=True)
ASSETS = SOURCE / 'Assets.xcassets'
ASSETS.mkdir(exist_ok=True)
META = {'author': 'xcode', 'version': 1}
(ASSETS / 'Contents.json').write_text(json.dumps({'info': META}, indent=2))
palette_root = ROOT.parent / 'brand-studio/dist/assets/palettes'
a = json.loads((palette_root / 'tokens-anatomie.json').read_text())
b = json.loads((palette_root / 'tokens-soufre.json').read_text())
roles = {'Canvas': 'canvas', 'Surface': 'surface', 'Ink': 'ink', 'Secondary': 'secondary', 'Line': 'line', 'AccentColor': 'accent', 'OnAccent': 'onaccent'}
colors = {name: (a[role], b[role]) for name, role in roles.items()}
colors.update({'Success': ('#245C3D', '#ADD7B5'), 'Restoration': ('#853E32', '#EDB7A8'), 'LaunchBackground': (a['canvas'], b['canvas'])})


def color(hex_value):
    return {'color-space': 'srgb', 'components': {**{name: f'{int(hex_value[i:i+2], 16)/255:.6f}' for name, i in [('red', 1), ('green', 3), ('blue', 5)]}, 'alpha': '1.000'}}


for name, (light, dark) in colors.items():
    directory = ASSETS / f'{name}.colorset'
    directory.mkdir(exist_ok=True)
    items = [{'idiom': 'universal', 'color': color(light)}, {'idiom': 'universal', 'appearances': [{'appearance': 'luminosity', 'value': 'dark'}], 'color': color(dark)}]
    if name in ['Secondary', 'Line']:
        items += [{'idiom': 'universal', 'appearances': [{'appearance': 'contrast', 'value': 'high'}], 'color': color(a['ink'])}, {'idiom': 'universal', 'appearances': [{'appearance': 'luminosity', 'value': 'dark'}, {'appearance': 'contrast', 'value': 'high'}], 'color': color(b['ink'])}]
    (directory / 'Contents.json').write_text(json.dumps({'colors': items, 'info': META}, indent=2))

ICON = ASSETS / 'AppIcon.appiconset'
ICON.mkdir(exist_ok=True)
(ICON / 'Contents.json').write_text(json.dumps({'images': [
    {'filename': 'AppIcon-anatomie.png', 'idiom': 'universal', 'platform': 'ios', 'size': '1024x1024'},
    {'filename': 'AppIcon-soufre.png', 'idiom': 'universal', 'platform': 'ios', 'size': '1024x1024', 'appearances': [{'appearance': 'luminosity', 'value': 'dark'}]},
], 'info': META}, indent=2))
svg = ET.parse(ROOT.parent / 'logo-explorations/organic/m-sillon-symbol-ink.svg').getroot()
group = next(child for child in svg if child.tag.endswith('g'))
shape = ''.join(ET.tostring(child, encoding='unicode') for child in group)
for palette in [a, b]:
    icon = '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">' + f'<rect width="1024" height="1024" fill="{palette["accent"]}"/><g transform="translate(128 128) scale(6)" color="{palette["onaccent"]}">{shape}</g></svg>'
    (ROOT / f'app-icon-{palette["id"]}.svg').write_text(icon)


def ident(label):
    return hashlib.sha1(label.encode()).hexdigest()[:24].upper()


objects = []


def obj(label, body):
    objects.append(f'\t\t{ident(label)} /* {label} */ = {{ {body} }};')
    return ident(label)


files = sorted(SOURCE.glob('*.swift'))
builds = []
references = []
for file in files:
    ref = obj(file.name, f'isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {file.name}; sourceTree = "<group>";')
    build = obj(file.name + ' in Sources', f'isa = PBXBuildFile; fileRef = {ref};')
    references.append(ref)
    builds.append(build)
asset_ref = obj('Assets.xcassets', 'isa = PBXFileReference; lastKnownFileType = folder.assetcatalog; path = Assets.xcassets; sourceTree = "<group>";')
asset_build = obj('Assets.xcassets in Resources', f'isa = PBXBuildFile; fileRef = {asset_ref};')
mascot_ref = obj('Lobbot', 'isa = PBXFileReference; lastKnownFileType = folder; path = Resources/Lobbot; sourceTree = "<group>";')
mascot_build = obj('Lobbot in Resources', f'isa = PBXBuildFile; fileRef = {mascot_ref};')
product = obj('LOBBOT.app', 'isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = LOBBOT.app; sourceTree = BUILT_PRODUCTS_DIR;')
sources_group = obj('LOBBOT', f'isa = PBXGroup; children = ({", ".join(references + [asset_ref, mascot_ref])},); path = LOBBOT; sourceTree = "<group>";')
products_group = obj('Products', f'isa = PBXGroup; children = ({product},); name = Products; sourceTree = "<group>";')
SECRETS = ROOT / 'Secrets.xcconfig'   # gitignored: holds GEMINI_API_KEY
if not SECRETS.exists():
    SECRETS.write_text((ROOT / 'Secrets.example.xcconfig').read_text())
secrets_ref = obj('Secrets.xcconfig', 'isa = PBXFileReference; lastKnownFileType = text.xcconfig; path = Secrets.xcconfig; sourceTree = "<group>";')
main_group = obj('Main group', f'isa = PBXGroup; children = ({secrets_ref}, {sources_group}, {products_group},); sourceTree = "<group>";')
sources_phase = obj('Sources', f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({", ".join(builds)},); runOnlyForDeploymentPostprocessing = 0;')
resources_phase = obj('Resources', f'isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({asset_build}, {mascot_build},); runOnlyForDeploymentPostprocessing = 0;')
frameworks_phase = obj('Frameworks', 'isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0;')

project_settings = '''ALWAYS_SEARCH_USER_PATHS = NO; CLANG_ENABLE_MODULES = YES; CLANG_ENABLE_OBJC_ARC = YES;
        IPHONEOS_DEPLOYMENT_TARGET = 17.0; SDKROOT = iphoneos; SWIFT_VERSION = 6.0;
        SWIFT_STRICT_CONCURRENCY = complete; SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor;
        SWIFT_APPROACHABLE_CONCURRENCY = YES; ENABLE_USER_SCRIPT_SANDBOXING = YES;'''
target_settings = '''ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon; ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
        CODE_SIGN_STYLE = Automatic; CURRENT_PROJECT_VERSION = 1; MARKETING_VERSION = 1.0;
        ENABLE_PREVIEWS = YES; GENERATE_INFOPLIST_FILE = YES; INFOPLIST_FILE = LOBBOT/Info.plist;
        INFOPLIST_KEY_CFBundleDisplayName = LOBBOT; INFOPLIST_KEY_LSApplicationCategoryType = "public.app-category.developer-tools";
        INFOPLIST_KEY_UIApplicationSceneManifest_Generation = YES; INFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents = YES;
        INFOPLIST_KEY_UILaunchScreen_Generation = YES; INFOPLIST_KEY_NSMicrophoneUsageDescription = "Dr. Lobbot listens when you talk to him."; INFOPLIST_KEY_UISupportedInterfaceOrientations = UIInterfaceOrientationPortrait;
        PRODUCT_BUNDLE_IDENTIFIER = dev.lobbot.prototype; PRODUCT_NAME = "$(TARGET_NAME)"; EXECUTABLE_NAME = LobbotApp;
        SUPPORTED_PLATFORMS = "iphoneos iphonesimulator"; TARGETED_DEVICE_FAMILY = 1;
        SUPPORTS_MACCATALYST = NO; SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD = NO;
        SWIFT_EMIT_LOC_STRINGS = YES;'''

project_configs, target_configs = [], []
for name in ['Debug', 'Release']:
    extra = 'DEBUG_INFORMATION_FORMAT = dwarf; SWIFT_OPTIMIZATION_LEVEL = "-Onone"; SWIFT_ACTIVE_COMPILATION_CONDITIONS = "DEBUG $(inherited)";' if name == 'Debug' else 'DEBUG_INFORMATION_FORMAT = "dwarf-with-dsym"; SWIFT_COMPILATION_MODE = wholemodule; SWIFT_OPTIMIZATION_LEVEL = "-O";'
    project_configs.append(obj('Project ' + name, f'isa = XCBuildConfiguration; buildSettings = {{ {project_settings} {extra} }}; name = {name};'))
    target_configs.append(obj('Target ' + name, f'isa = XCBuildConfiguration; baseConfigurationReference = {secrets_ref}; buildSettings = {{ {target_settings} }}; name = {name};'))
project_list = obj('Project configuration list', f'isa = XCConfigurationList; buildConfigurations = ({", ".join(project_configs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target_list = obj('Target configuration list', f'isa = XCConfigurationList; buildConfigurations = ({", ".join(target_configs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target = obj('LOBBOT target', f'isa = PBXNativeTarget; buildConfigurationList = {target_list}; buildPhases = ({sources_phase}, {frameworks_phase}, {resources_phase},); buildRules = (); dependencies = (); name = LOBBOT; productName = LOBBOT; productReference = {product}; productType = "com.apple.product-type.application";')
root_id = obj('Project object', f'isa = PBXProject; attributes = {{ BuildIndependentTargetsInParallel = 1; LastSwiftUpdateCheck = 2600; LastUpgradeCheck = 2600; TargetAttributes = {{ {target} = {{ CreatedOnToolsVersion = 26.0; ProvisioningStyle = Automatic; }}; }}; }}; buildConfigurationList = {project_list}; compatibilityVersion = "Xcode 14.0"; developmentRegion = fr; hasScannedForEncodings = 0; knownRegions = (fr, en, Base,); mainGroup = {main_group}; productRefGroup = {products_group}; projectDirPath = ""; projectRoot = ""; targets = ({target},);')
(PROJECT / 'project.pbxproj').write_text('// !$*UTF8*$!\n{\n\tarchiveVersion = 1;\n\tclasses = {};\n\tobjectVersion = 56;\n\tobjects = {\n' + '\n'.join(objects) + f'\n\t}};\n\trootObject = {root_id};\n}}\n')
workspace = PROJECT / 'project.xcworkspace'
workspace.mkdir(exist_ok=True)
(workspace / 'contents.xcworkspacedata').write_text('<?xml version="1.0" encoding="UTF-8"?><Workspace version="1.0"><FileRef location="self:"></FileRef></Workspace>')
scheme_dir = PROJECT / 'xcshareddata/xcschemes'
scheme_dir.mkdir(parents=True, exist_ok=True)
reference = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="LOBBOT.app" BlueprintName="LOBBOT" ReferencedContainer="container:LOBBOT.xcodeproj"/>'
for name, mode in [('LOBBOT', None), ('LOBBOT-A-Anatomie', 'anatomie'), ('LOBBOT-B-Soufre', 'soufre')]:
    arguments = '' if mode is None else f'<CommandLineArguments><CommandLineArgument argument="-lobbot.brandMode" isEnabled="YES"/><CommandLineArgument argument="{mode}" isEnabled="YES"/></CommandLineArguments>'
    scheme = f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2600" version="1.7">
<BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{reference}</BuildActionEntry></BuildActionEntries></BuildAction>
<TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables/></TestAction>
<LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable>{arguments}</LaunchAction>
<ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></ProfileAction>
<AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>'''
    ET.fromstring(scheme)
    (scheme_dir / f'{name}.xcscheme').write_text(scheme)
print(f'Created {PROJECT.name}: {len(files)} Swift sources, local mascot folder, semantic color assets, three shared schemes.')
