@echo ON

set SolutionDir=%1
set ProjectDir=%2
set ConfigurationName=%3

cd "%SolutionDir%..\cs_interop"
cargo build 

set SOURCE_DLL_FILENAME="%SolutionDir%..\target\debug\cs_interop.dll"
set TARGET_DLL_FILENAME="%ProjectDir%core_interop.dll"

echo F | xcopy %SOURCE_DLL_FILENAME% %TARGET_DLL_FILENAME% /D /Y
