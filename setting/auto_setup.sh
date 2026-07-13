#!/bin/bash
set -euo pipefail

########################################
# 기본 설정
########################################
ENV_YML="$(pwd)/environment.yml"
USER_NAME="$(whoami)"
HOME_BASE="/home1/${USER_NAME}"

NEC_BASE="${HOME_BASE}/NEC"
GIT_BASE="${NEC_BASE}/git"
PYAEDT_REPO="${GIT_BASE}/pyaedt_library"

MFT_PARENT="${NEC_BASE}/MFT_1MW"
MFT_REPO="${MFT_PARENT}/MFT_1MW_2026"

PYAEDT_GIT_URL="https://github.com/Schwalbe262/pyaedt_library"
PYAEDT_BRANCH="pyaedt_022"
MFT_GIT_URL="https://github.com/Schwalbe262/MFT_1MW_2026"

MINICONDA_DIR="${HOME}/miniconda3"
ANACONDA_DIR="${HOME}/anaconda3"

########################################
# 공통 함수
########################################
log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

########################################
# 사전 체크
########################################
if [ ! -f "$ENV_YML" ]; then
    log_error "현재 디렉토리에 environment.yml 이 없습니다: $ENV_YML"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    log_error "git 명령어를 찾을 수 없습니다."
    exit 1
fi

########################################
# conda 로드 함수
########################################
load_conda() {
    # 1) miniconda 경로 직접 source
    if [ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
        return 0
    fi

    # 2) anaconda 경로 직접 source
    if [ -f "${ANACONDA_DIR}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${ANACONDA_DIR}/etc/profile.d/conda.sh"
        return 0
    fi

    # 3) PATH에 있으면 hook 사용
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi

    return 1
}

########################################
# miniconda 설치 함수
########################################
install_miniconda() {
    log_info "conda를 찾지 못했습니다. Miniconda 설치를 시도합니다."

    if [ -d "$MINICONDA_DIR" ]; then
        log_error "'$MINICONDA_DIR' 디렉토리는 이미 존재하는데 conda.sh를 찾지 못했습니다."
        log_error "기존 설치가 불완전하거나 깨졌을 가능성이 있습니다."
        log_error "확인 경로: $MINICONDA_DIR"
        log_error "필요하면 아래 중 하나를 선택하세요."
        log_error "1) 기존 ~/miniconda3를 삭제 후 다시 실행"
        log_error "2) 수동으로 설치 상태 점검"
        exit 1
    fi

    ARCH="$(uname -m)"
    OS="$(uname -s)"

    if [ "$OS" != "Linux" ]; then
        log_error "이 스크립트는 Linux 기준입니다. 현재 OS: $OS"
        exit 1
    fi

    case "$ARCH" in
        x86_64)
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            ;;
        aarch64|arm64)
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            ;;
        *)
            log_error "지원하지 않는 아키텍처입니다: $ARCH"
            exit 1
            ;;
    esac

    INSTALLER="/tmp/miniconda_installer_${USER_NAME}.sh"

    if command -v curl >/dev/null 2>&1; then
        log_info "curl로 Miniconda 설치 파일 다운로드"
        curl -fsSL "$MINICONDA_URL" -o "$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
        log_info "wget으로 Miniconda 설치 파일 다운로드"
        wget -O "$INSTALLER" "$MINICONDA_URL"
    else
        log_error "curl 또는 wget 이 필요합니다."
        exit 1
    fi

    log_info "Miniconda 설치: $MINICONDA_DIR"
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$INSTALLER"

    if [ ! -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
        log_error "Miniconda 설치 후 conda.sh를 찾지 못했습니다."
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
}

########################################
# conda 준비
########################################
if load_conda; then
    log_info "기존 conda 설치를 로드했습니다."
else
    install_miniconda
    if ! load_conda; then
        log_error "Miniconda 설치 후에도 conda 로드에 실패했습니다."
        exit 1
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    log_error "conda 명령어를 사용할 수 없습니다."
    exit 1
fi

log_info "conda executable: $(command -v conda)"
log_info "conda version   : $(conda --version)"

########################################
# bashrc 등록
########################################
if [ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
    if ! grep -qxF 'source ~/miniconda3/etc/profile.d/conda.sh' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> "${HOME}/.bashrc"
        log_info "~/.bashrc 에 miniconda source 구문 추가 완료"
    else
        log_info "~/.bashrc 에 miniconda source 구문이 이미 존재함"
    fi
elif [ -f "${ANACONDA_DIR}/etc/profile.d/conda.sh" ]; then
    if ! grep -qxF 'source ~/anaconda3/etc/profile.d/conda.sh' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'source ~/anaconda3/etc/profile.d/conda.sh' >> "${HOME}/.bashrc"
        log_info "~/.bashrc 에 anaconda source 구문 추가 완료"
    else
        log_info "~/.bashrc 에 anaconda source 구문이 이미 존재함"
    fi
fi

########################################
# environment.yml 에서 환경 이름 추출
########################################
ENV_NAME="$(awk '/^name:[[:space:]]*/{print $2; exit}' "$ENV_YML")"

if [ -z "${ENV_NAME:-}" ]; then
    log_error "environment.yml 에 name: 항목이 없습니다."
    exit 1
fi

log_info "Conda environment name: $ENV_NAME"

########################################
# conda 환경 생성
########################################
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    log_info "conda 환경 '$ENV_NAME' 이미 존재함. 생성 건너뜀."
else
    log_info "conda 환경 '$ENV_NAME' 생성 중..."
    conda env create -f "$ENV_YML"
fi

########################################
# 디렉토리 생성
########################################
mkdir -p "$GIT_BASE"
mkdir -p "$MFT_PARENT"

########################################
# pyaedt_library clone 또는 update
########################################
if [ -d "$PYAEDT_REPO/.git" ]; then
    log_info "pyaedt_library 이미 존재. 업데이트 수행."
    git -C "$PYAEDT_REPO" fetch origin
    git -C "$PYAEDT_REPO" checkout "$PYAEDT_BRANCH"
    git -C "$PYAEDT_REPO" pull origin "$PYAEDT_BRANCH"
else
    log_info "pyaedt_library clone 시작"
    git clone -b "$PYAEDT_BRANCH" "$PYAEDT_GIT_URL" "$PYAEDT_REPO"
fi

########################################
# MFT_1MW_2026 clone 또는 update
########################################
if [ -d "$MFT_REPO/.git" ]; then
    log_info "MFT_1MW_2026 이미 존재. 업데이트 수행."
    git -C "$MFT_REPO" fetch origin
    git -C "$MFT_REPO" pull origin
else
    log_info "MFT_1MW_2026 clone 시작"
    git clone "$MFT_GIT_URL" "$MFT_REPO"
fi

########################################
# 완료 메시지
########################################
echo
echo "[DONE] 설정 완료"
echo "USER        : $USER_NAME"
echo "HOME_BASE   : $HOME_BASE"
echo "Conda env   : $ENV_NAME"
echo "pyaedt repo : $PYAEDT_REPO"
echo "pyaedt src  : ${PYAEDT_REPO}/src/"
echo "MFT repo    : $MFT_REPO"