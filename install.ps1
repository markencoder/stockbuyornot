& "C:\ProgramData\Anaconda3\envs\tower312\python.exe" -m pip install akshare pytest streamlit==1.40.2
& "C:\ProgramData\Anaconda3\envs\tower312\python.exe" -m pip install --no-build-isolation -e .
& "C:\ProgramData\Anaconda3\envs\tower312\python.exe" -c "import akshare as ak; import streamlit; import stockbuyornot; print('akshare', ak.__version__); print('streamlit', streamlit.__version__); print('stockbuyornot', stockbuyornot.__version__)"
