{
  description = "PSI development shells";

  inputs = {
    self.submodules = true;
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      lib = pkgs.lib;
      runtimeLib = import ./nix/lib/runtime.nix {
        inherit lib pkgs;
      };
      psiBaseRuntime = import ./nix/runtime-base.nix { inherit pkgs; };
      hostGpuRuntime = import ./nix/runtime-host-gpu.nix { inherit pkgs; };

      commonPackages = [
        pkgs.bashInteractive
        pkgs.coreutils
        pkgs.curl
        pkgs.git
        pkgs.git-lfs
        pkgs.uv
        pkgs.wget
      ];

      commonShellHook = ''
        if [ -f "$HOME/.env" ]; then
          set -a
          . "$HOME/.env"
          set +a
        fi

        _psi_detect_torch_cuda_arch_list() {
          if [ -n "''${TORCH_CUDA_ARCH_LIST:-}" ]; then
            return 0
          fi

          if ! command -v nvidia-smi >/dev/null 2>&1; then
            return 1
          fi

          _psi_compute_caps="$(
            nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
              | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
              | sed '/^$/d'
          )"

          if [ -z "$_psi_compute_caps" ]; then
            return 1
          fi

          _psi_compute_cap_count="$(printf '%s\n' "$_psi_compute_caps" | sort -u | wc -l)"
          if [ "$_psi_compute_cap_count" -ne 1 ]; then
            return 1
          fi

          _psi_compute_cap="$(printf '%s\n' "$_psi_compute_caps" | head -n 1)"
          if [ -n "$_psi_compute_cap" ]; then
            export TORCH_CUDA_ARCH_LIST="$_psi_compute_cap+PTX"
            return 0
          fi

          return 1
        }

        _psi_detect_torch_cuda_arch_list || true
        unset -f _psi_detect_torch_cuda_arch_list
      '';
    in {
      devShells.${system} = {
        default = runtimeLib.mkRuntimeShell {
          name = "psi";
          runtimes = [ psiBaseRuntime hostGpuRuntime ];
          extraPackages = commonPackages;
          extraShellHook = ''
            ${commonShellHook}
            echo "SIMPLE runtime composed into root shell"
          '';
          stdenv = pkgs.gcc13Stdenv;
        };

        psi = runtimeLib.mkRuntimeShell {
          name = "psi";
          runtimes = [ psiBaseRuntime hostGpuRuntime ];
          extraPackages = commonPackages;
          extraShellHook = commonShellHook;
          stdenv = pkgs.gcc13Stdenv;
        };

      };
    };
}
