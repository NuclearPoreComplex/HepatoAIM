@echo off
chcp 65001
setlocal enabledelayedexpansion

rem 获取脚本所在目录的路径
set "script_dir=%~dp0"

rem 遍历当前目录下的所有文件夹
for /d %%i in (*) do (
    rem 检查文件夹名称是否符合 *_* 的模式
    echo %%i | findstr /r ".*_.*" >nul
    if !errorlevel! equ 0 (
        echo 正在处理文件夹: %%i
        pushd %%i
        rem 复制 gmx_p1.bat、topol.py 和其他配置文件到子目录
        copy "%script_dir%gmx_p1.bat" . /y
        copy "%script_dir%topol.py" . /y
        copy "%script_dir%make_ndx.txt" . /y
        copy "%script_dir%em.mdp" . /y
        copy "%script_dir%md.mdp" . /y
        copy "%script_dir%npt.mdp" . /y
        copy "%script_dir%nvt.mdp" . /y
        rem 在子目录中运行 gmx_p1.bat
        echo. | call gmx_p1.bat
        echo %%i部分步骤执行完毕！
        popd
    )
)
echo 所有步骤执行完毕！
pause