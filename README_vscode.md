## Remote Debug Integration

0.  `pip install ptvsd==3.0.0` in local machine (in the python environment used by vscode)

1.  `pip install ptvsd==3.0.0` on ros machine

1.  Add entry like so to ~/.ssh/config with the hostname of your ros node:

```
Host rosvm
    HostName 172.16.5.129
    User student
    LocalForward 11000 localhost:11000
    LocalForward 11001 localhost:11001
    LocalForward 11002 localhost:11002
    LocalForward 11003 localhost:11003
    LocalForward 11004 localhost:11004
    LocalForward 11005 localhost:11005
```

3.  Ensure correct remoteRoot config in ~/.vscode/launch.json

```
    "remoteRoot": "/home/student/vmshared/CarND-Capstone", // Path to the program location on the remote computer.
```

4.  Uncomment lines in python file that you want to debug

```
# import ptvsd
# ptvsd.enable_attach("ros_secret", address=('127.0.0.1', 11000))
# ptvsd.wait_for_attach()
```

5.  Invoke roslaunch -- all the python processes will start, the ones with ptsvd.wait_for_attach() will block until the vscode debugger is connected

6.  Attach to desired process in vscode
