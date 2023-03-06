{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = import nixpkgs { inherit system; config.allowUnfree = true; }; in with pkgs;{
          devShells.default = mkShell {
            nativeBuildInputs = [
              rustPlatform.rust.cargo
              rustPlatform.rust.rustc
              rustPlatform.bindgenHook
              rust-analyzer
              rustfmt
            ];
            RUSTC_BOOTSTRAP = 1;
          };
        }
      );
}
