#!/bin/bash
################################################################################
# 文件简单重命名脚本
# 保持扁平结构，只重命名文件和移动旧代码到archive
################################################################################

cd "$(dirname "$0")"

echo "=========================================="
echo "文件整理和重命名"
echo "=========================================="
echo ""

# 检查是否已备份
if [ ! -d "../dc_backup" ]; then
    echo "⚠️  建议先备份！执行："
    echo "   cd /c/Users/HP/Desktop && cp -r dc dc_backup"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

echo "=== 第1步: 重命名主程序 ==="
if [ -f "DtestStaticSPAINew30_fixed.cpp" ]; then
    mv DtestStaticSPAINew30_fixed.cpp spai_single_dcu.cpp
    echo "✓ DtestStaticSPAINew30_fixed.cpp -> spai_single_dcu.cpp"
fi

if [ -f "DtestStaticSPAINew30_multiDCU.cpp" ]; then
    mv DtestStaticSPAINew30_multiDCU.cpp spai_multi_dcu.cpp
    echo "✓ DtestStaticSPAINew30_multiDCU.cpp -> spai_multi_dcu.cpp"
fi
echo ""

echo "=== 第2步: 重命名文档 ==="
if [ -f "README_DCU.md" ]; then
    mv README_DCU.md README.md
    echo "✓ README_DCU.md -> README.md"
fi

if [ -f "DtestStaticSPAINew30_fixed_CHANGELOG.md" ]; then
    mv DtestStaticSPAINew30_fixed_CHANGELOG.md CHANGELOG_SINGLE_DCU.md
    echo "✓ DtestStaticSPAINew30_fixed_CHANGELOG.md -> CHANGELOG_SINGLE_DCU.md"
fi

if [ -f "MULTI_DCU_GUIDE.md" ]; then
    mv MULTI_DCU_GUIDE.md GUIDE_MULTI_DCU.md
    echo "✓ MULTI_DCU_GUIDE.md -> GUIDE_MULTI_DCU.md"
fi

if [ -f "MULTI_DCU_SUMMARY.md" ]; then
    mv MULTI_DCU_SUMMARY.md SUMMARY_MULTI_DCU.md
    echo "✓ MULTI_DCU_SUMMARY.md -> SUMMARY_MULTI_DCU.md"
fi

if [ -f "MULTI_DCU_QUICKREF.md" ]; then
    mv MULTI_DCU_QUICKREF.md QUICKREF_MULTI_DCU.md
    echo "✓ MULTI_DCU_QUICKREF.md -> QUICKREF_MULTI_DCU.md"
fi
echo ""

echo "=== 第3步: 重命名编译配置 ==="
if [ -f "Makefile.dcu" ]; then
    mv Makefile.dcu Makefile.single
    echo "✓ Makefile.dcu -> Makefile.single"
fi

if [ -f "Makefile.multiDCU" ]; then
    mv Makefile.multiDCU Makefile.multi
    echo "✓ Makefile.multiDCU -> Makefile.multi"
fi

if [ -f "build_dcu.sh" ]; then
    mv build_dcu.sh build_single.sh
    chmod +x build_single.sh
    echo "✓ build_dcu.sh -> build_single.sh"
fi
echo ""

echo "=== 第4步: 重命名求解器头文件 ==="
if [ -f "bicgstab/cublas2_csr_pbicgstab_12.2.h" ]; then
    mv bicgstab/cublas2_csr_pbicgstab_12.2.h bicgstab/bicgstab_solver.h
    echo "✓ cublas2_csr_pbicgstab_12.2.h -> bicgstab_solver.h"
fi
echo ""

echo "=== 第5步: 归档旧代码 ==="
mkdir -p archive
if [ -f "DtestStaticSPAINew30.cpp" ]; then
    mv DtestStaticSPAINew30.cpp archive/
    echo "✓ DtestStaticSPAINew30.cpp -> archive/"
fi

if [ -f "DtestStaticSPAINew30.cu" ]; then
    mv DtestStaticSPAINew30.cu archive/
    echo "✓ DtestStaticSPAINew30.cu -> archive/"
fi

if [ -f "duospai.cpp" ]; then
    mv duospai.cpp archive/
    echo "✓ duospai.cpp -> archive/"
fi
echo ""

echo "=== 第6步: 整理工具 ==="
mkdir -p tools
if [ -f "chaxun.cpp" ]; then
    mv chaxun.cpp tools/check_dcu.cpp
    echo "✓ chaxun.cpp -> tools/check_dcu.cpp"
fi
echo ""

echo "=== 第7步: 整理示例 ==="
mkdir -p examples
if [ -f "demo.cpp" ]; then
    mv demo.cpp examples/vector_add_demo.cpp
    echo "✓ demo.cpp -> examples/vector_add_demo.cpp"
fi
echo ""

echo "=== 第8步: 删除编译产物 ==="
if [ -f "DtestStaticSPAINew30" ]; then
    rm -f DtestStaticSPAINew30
    echo "✓ 已删除: DtestStaticSPAINew30 (编译产物)"
fi
echo ""

echo "=== 第9步: 删除无用Makefile ==="
if [ -f "common/Makefile" ]; then
    rm -f common/Makefile
    echo "✓ 已删除: common/Makefile"
fi

if [ -f "common/Makefile1070" ]; then
    rm -f common/Makefile1070
    echo "✓ 已删除: common/Makefile1070"
fi
echo ""

echo "=========================================="
echo "文件整理完成！"
echo "=========================================="
echo ""
echo "主要文件："
echo "  ✓ spai_single_dcu.cpp     - 单DCU主程序"
echo "  ✓ spai_multi_dcu.cpp      - 多DCU主程序"
echo "  ✓ README.md               - 主文档"
echo ""
echo "编译："
echo "  ✓ Makefile.single         - 单DCU编译"
echo "  ✓ Makefile.multi          - 多DCU编译"
echo "  ✓ build_single.sh         - 编译脚本"
echo ""
echo "归档目录："
echo "  ✓ archive/                - 旧代码"
echo "  ✓ tools/                  - 工具"
echo "  ✓ examples/               - 示例"
echo ""
echo "需要更新的文件（包含路径）："
echo "  ⚠️  spai_single_dcu.cpp   - 修改 #include \"bicgstab/cublas2...\" 为 \"bicgstab/bicgstab_solver.h\""
echo "  ⚠️  spai_multi_dcu.cpp    - 同上"
echo "  ⚠️  Makefile.single       - 如需要"
echo "  ⚠️  Makefile.multi        - 如需要"
echo ""
