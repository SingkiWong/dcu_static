#!/bin/bash
################################################################################
# 更新代码中的文件路径引用
# 在重命名后执行此脚本
################################################################################

cd "$(dirname "$0")"

echo "=========================================="
echo "更新代码中的路径引用"
echo "=========================================="
echo ""

echo "=== 更新 spai_single_dcu.cpp ==="
if [ -f "spai_single_dcu.cpp" ]; then
    # 备份
    cp spai_single_dcu.cpp spai_single_dcu.cpp.bak

    # 更新bicgstab头文件引用
    sed -i 's|bicgstab/cublas2_csr_pbicgstab_12\.2\.h|bicgstab/bicgstab_solver.h|g' spai_single_dcu.cpp

    echo "✓ 已更新 bicgstab 头文件引用"
    echo "  备份: spai_single_dcu.cpp.bak"
else
    echo "⚠️  文件不存在，请先运行 rename_simple.sh"
fi
echo ""

echo "=== 更新 spai_multi_dcu.cpp ==="
if [ -f "spai_multi_dcu.cpp" ]; then
    # 备份
    cp spai_multi_dcu.cpp spai_multi_dcu.cpp.bak

    # 更新bicgstab头文件引用
    sed -i 's|bicgstab/cublas2_csr_pbicgstab_12\.2\.h|bicgstab/bicgstab_solver.h|g' spai_multi_dcu.cpp

    echo "✓ 已更新 bicgstab 头文件引用"
    echo "  备份: spai_multi_dcu.cpp.bak"
else
    echo "⚠️  文件不存在，请先运行 rename_simple.sh"
fi
echo ""

echo "=== 更新 Makefile.single ==="
if [ -f "Makefile.single" ]; then
    # 备份
    cp Makefile.single Makefile.single.bak

    # 更新可执行文件名
    sed -i 's|DtestStaticSPAINew30_fixed|spai_single_dcu|g' Makefile.single

    echo "✓ 已更新可执行文件名"
    echo "  备份: Makefile.single.bak"
else
    echo "⚠️  文件不存在"
fi
echo ""

echo "=== 更新 Makefile.multi ==="
if [ -f "Makefile.multi" ]; then
    # 备份
    cp Makefile.multi Makefile.multi.bak

    # 更新可执行文件名
    sed -i 's|DtestStaticSPAINew30_multiDCU|spai_multi_dcu|g' Makefile.multi
    sed -i 's|DtestStaticSPAINew30_fixed|spai_single_dcu|g' Makefile.multi

    echo "✓ 已更新可执行文件名"
    echo "  备份: Makefile.multi.bak"
else
    echo "⚠️  文件不存在"
fi
echo ""

echo "=== 更新 build_single.sh ==="
if [ -f "build_single.sh" ]; then
    # 备份
    cp build_single.sh build_single.sh.bak

    # 更新可执行文件名和Makefile引用
    sed -i 's|DtestStaticSPAINew30_fixed|spai_single_dcu|g' build_single.sh
    sed -i 's|Makefile\.dcu|Makefile.single|g' build_single.sh

    echo "✓ 已更新脚本内容"
    echo "  备份: build_single.sh.bak"
else
    echo "⚠️  文件不存在"
fi
echo ""

echo "=========================================="
echo "路径更新完成！"
echo "=========================================="
echo ""
echo "已更新的文件："
echo "  ✓ spai_single_dcu.cpp"
echo "  ✓ spai_multi_dcu.cpp"
echo "  ✓ Makefile.single"
echo "  ✓ Makefile.multi"
echo "  ✓ build_single.sh"
echo ""
echo "备份文件（.bak）已创建，如有问题可恢复"
echo ""
echo "下一步："
echo "  1. 测试编译: make -f Makefile.single"
echo "  2. 或使用脚本: ./build_single.sh"
echo ""
