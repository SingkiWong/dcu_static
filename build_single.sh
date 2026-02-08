#!/bin/bash

################################################################################
# DCU 编译脚本 - Static SPAI Preconditioner
# 快速编译和测试工具
################################################################################

set -e  # 出错时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo -e "DCU Static SPAI 编译脚本"
echo -e "==========================================${NC}"

# 检查ROCm环境
if [ -z "$ROCM_PATH" ]; then
    export ROCM_PATH=/opt/rocm
    echo -e "${YELLOW}ROCM_PATH未设置，使用默认值: $ROCM_PATH${NC}"
fi

# 检查hipcc是否可用
if ! command -v hipcc &> /dev/null; then
    echo -e "${RED}错误: 未找到hipcc编译器${NC}"
    echo -e "${YELLOW}请确保ROCm已正确安装并设置PATH${NC}"
    echo -e "  export PATH=\$ROCM_PATH/bin:\$PATH"
    exit 1
fi

echo -e "${GREEN}✓ 找到hipcc编译器${NC}"
hipcc --version | head -n1

# 检查DCU设备
echo ""
echo -e "${BLUE}检查DCU设备...${NC}"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
else
    echo -e "${YELLOW}警告: rocm-smi不可用，无法查询设备信息${NC}"
fi

# 自动检测GPU架构
echo ""
echo -e "${BLUE}检测GPU架构...${NC}"
if command -v rocminfo &> /dev/null; then
    GPU_NAME=$(rocminfo | grep "Name:" | head -n1 | awk '{print $2}')
    echo -e "检测到GPU: ${GREEN}$GPU_NAME${NC}"

    # 根据GPU名称推断架构
    case "$GPU_NAME" in
        *"gfx906"*)
            GPU_ARCH="gfx906"
            echo -e "架构: ${GREEN}gfx906 (MI50/MI60/海光Z100)${NC}"
            ;;
        *"gfx908"*)
            GPU_ARCH="gfx908"
            echo -e "架构: ${GREEN}gfx908 (MI100)${NC}"
            ;;
        *"gfx90a"*)
            GPU_ARCH="gfx90a"
            echo -e "架构: ${GREEN}gfx90a (MI210/MI250)${NC}"
            ;;
        *)
            GPU_ARCH="gfx906"
            echo -e "${YELLOW}未能识别GPU架构，使用默认值: gfx906${NC}"
            ;;
    esac
else
    GPU_ARCH="gfx906"
    echo -e "${YELLOW}rocminfo不可用，使用默认架构: gfx906${NC}"
fi

# 允许用户覆盖架构
if [ ! -z "$1" ]; then
    GPU_ARCH="$1"
    echo -e "${BLUE}使用用户指定的架构: $GPU_ARCH${NC}"
fi

# 开始编译
echo ""
echo -e "${BLUE}=========================================="
echo -e "开始编译..."
echo -e "==========================================${NC}"

make -f Makefile.single GPU_ARCH=$GPU_ARCH clean
make -f Makefile.single GPU_ARCH=$GPU_ARCH

# 编译成功
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo -e "✓ 编译成功！"
    echo -e "==========================================${NC}"
    echo -e "${GREEN}可执行文件: spai_single_dcu${NC}"
    echo ""
    echo -e "${BLUE}运行方法:${NC}"
    echo -e "  ./spai_single_dcu"
    echo -e "  然后输入矩阵文件名 (例如: circuit_2.mtx)"
    echo ""
    echo -e "${BLUE}或使用测试命令:${NC}"
    echo -e "  make -f Makefile.single test"
    echo ""
else
    echo -e "${RED}=========================================="
    echo -e "✗ 编译失败"
    echo -e "==========================================${NC}"
    exit 1
fi
