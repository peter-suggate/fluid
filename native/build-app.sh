#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
swift build -c release

app="$PWD/.build/Fluid Lab Metal.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp .build/release/FluidMetal "$app/Contents/MacOS/FluidMetal"
cp Info.plist "$app/Contents/Info.plist"
cp -R .build/release/FluidMetal_FluidMetal.bundle "$app/Contents/Resources/"
codesign --force --sign - --deep "$app"
echo "$app"
