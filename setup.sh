echo "Installing basic (required) modules ... "
sudo pip3 install wheel
sudo pip3 install setuptools

echo "Installing DroidAutoML's Python3 specific requirements ... "
sudo python3 setup_python.py bdist_wheel
sudo pip3 install dist/droidautoml-1.0.0-py3-none-any.whl

