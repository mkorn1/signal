# Generate the binary for x86_64 architecture
swiftc -target x86_64-apple-macos10.15 -o AuthSession-x86_64 ./src/AuthSession.swift

# Generate the binary for arm64 architecture
swiftc -target arm64-apple-macos11 -o AuthSession-arm64 ./src/AuthSession.swift

# Use the lipo command to create a universal binary
lipo -create -output ./resources/AuthSession AuthSession-x86_64 AuthSession-arm64

# Verify the universal binary
file ./resources/AuthSession

rm AuthSession-x86_64 AuthSession-arm64
