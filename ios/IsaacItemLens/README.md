# Isaac Item Lens for iOS

Native SwiftUI client for fully offline Isaac object detection and retrieval.

## Generate and open

```bash
cd ios/IsaacItemLens
./sync_resources.sh
xcodegen generate
open IsaacItemLens.xcodeproj
```

The project consumes the Core ML packages from `../ModelConversion/output` and
bundles the 943-object encyclopedia, icons, and Float16 vector index. Select a
development team under Signing & Capabilities before running on a physical
iPhone.

Run `sync_resources.sh` again whenever the encyclopedia, icons, or vector index
changes. The Core ML packages are referenced directly from
`../ModelConversion/output` and are recompiled by Xcode automatically.
