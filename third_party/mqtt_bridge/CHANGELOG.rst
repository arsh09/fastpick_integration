^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package mqtt_bridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------

1.4.0 (2021-11-19 13:31)
------------------------
* empty changelog
* changlogs
* changelogs
* changelogs
* changelogs
* Contributors: Marc Hanheide

1.3.0 (2021-11-19 12:07)
------------------------
* updated changelogs
* Merge pull request `#12 <https://github.com/LCAS/mqtt_bridge/issues/12>`_ from LCAS/reconnection
  Implementing stable re-connection
* Merge pull request `#13 <https://github.com/LCAS/mqtt_bridge/issues/13>`_ from LCAS/marc-hanheide-patch-1
  attempt to use new L-CAS tooling
* remove noetic
* remove noetic
* remove noetic
* correct tooling
* attempt to use new L-CAS tooling
* change keep alive to 5 seconds to detect earlier
* deal with unstable network connection
  * ensure loop continues even after network failure
  * set session to NOT clean and use client_id (to maintain subscription)
  * set corresponding ROS params to ensure we always have a valid client_id and a not clean session
* don't retain control messages (conflict on reconnect)
* WIP: working towards reconnection
* included some working rostest
* renamed
* Merge pull request `#11 <https://github.com/LCAS/mqtt_bridge/issues/11>`_ from LCAS/github_workflow
  first workflow
* compelted workflowx
* first workflow
* Contributors: LCAS build farm, Marc Hanheide

1.2.2 (2021-03-11)
------------------
* updated changelogs
* Merge pull request `#4 <https://github.com/LCAS/mqtt_bridge/issues/4>`_ from LCAS/futures
  simple threading solution
* simple threading solution
* Contributors: LCAS build farm, Marc Hanheide

1.2.1 (2021-03-10)
------------------
* changelogs
* Merge pull request `#2 <https://github.com/LCAS/mqtt_bridge/issues/2>`_ from francescodelduchetto/avoid-loops
  Avoid loops in connection with the server
* do not start a new bridge in the Server if the bridge with the reverse connection is already running bcs it create loops in ros messages
* Update README.md
* improved documentation
* Contributors: Marc Hanheide, francescodelduchetto

1.2.0 (2021-01-30 20:29)
------------------------
* changelogs
* added LocalServiceProxy
* renamed to RemoteService
* Contributors: Marc Hanheide

1.1.0 (2021-01-30 19:29)
------------------------
* changelogs
* Merge pull request `#1 <https://github.com/LCAS/mqtt_bridge/issues/1>`_ from LCAS/remoteserver
  Remoteserver implementation complete
* implementation of RemoteServer working
* WIP: towards a remote server call
* Contributors: Marc Hanheide

1.0.1 (2021-01-29 18:54)
------------------------
* changelog
* missing install target
* Merge branch 'pr_upstream' into melodic-devel
* example now also has DynamicServer example
  and further examples of qos and latch
* action server launch file is working
  with prefix and dynamic server
* DynamicServer implementation working
* added launching own MQTT server
* removed
* intial actionlib mqtt setup
* support for latched (retained) topics and QoS
* Contributors: Marc Hanheide

1.0.0 (2021-01-29 18:14)
------------------------
* changelog
* 0.2.1
* changelog
* changed version number
* example now also has DynamicServer example
  and further examples of qos and latch
* action server launch file is working
  with prefix and dynamic server
* DynamicServer implementation working
* added launching own MQTT server
* removed
* intial actionlib mqtt setup
* support for latched (retained) topics and QoS
* ignore vscode
* changed maintainer
* fix install target for directories
* Contributors: Marc Hanheide

0.1.8 (2020-12-31)
------------------
* update CHANGELOG
* Merge pull request `#41 <https://github.com/LCAS/mqtt_bridge/issues/41>`_ from groove-x/feature/rostest
  configure rostest and run it on circleci
* configure add_custom_target. (close `#22 <https://github.com/LCAS/mqtt_bridge/issues/22>`_)
* run rostest on circleci
* add rostest
* add comment about msg_type (close `#20 <https://github.com/LCAS/mqtt_bridge/issues/20>`_) (`#39 <https://github.com/LCAS/mqtt_bridge/issues/39>`_)
* Merge pull request `#38 <https://github.com/LCAS/mqtt_bridge/issues/38>`_ from groove-x/feature/dev-requirements
  split requirements.txt
* split requirements
* Merge pull request `#37 <https://github.com/LCAS/mqtt_bridge/issues/37>`_ from groove-x/feature/unittest
  add unittests
* configure circleci
* add unittests
* Merge pull request `#32 <https://github.com/LCAS/mqtt_bridge/issues/32>`_ from Roboterbastler/js/fix_setup_warning_install_requires
  Fix 'install_requires' warning when building with --install
* Fix 'install_requires' warning when building with --install
  distutils apparently doesn't support 'install_requires', replacing it with setuptools (which is also matching the example in http://docs.ros.org/melodic/api/catkin/html/howto/format2/installing_python.html) fixes that
* Contributors: Jacob Seibert, Junya Hayashi

0.1.7 (2020-06-21)
------------------
* Merge pull request `#27 <https://github.com/LCAS/mqtt_bridge/issues/27>`_ from kapilPython/prpth-fix-branch
  mqtt subscribing to private path had a small bug
* Update src/mqtt_bridge/mqtt_client.py
  rospy.logdebug command could be removed this was just added to debug the changes.
  Co-authored-by: Yuma Mihira <yuma-m@users.noreply.github.com>
* mqtt subscribing to private path had a small bug
* Merge pull request `#26 <https://github.com/LCAS/mqtt_bridge/issues/26>`_ from 5tan/patch-1
  Fixed bug in frequency limit
* Fixed bug in frequency limit
  Fixed wrong condition check. Now it works the same as in `MqttToRosBridge`.
* Merge pull request `#10 <https://github.com/LCAS/mqtt_bridge/issues/10>`_ from groove-x/hotfix/add-bson
  Add bson module in requirements.txt
* Update README.md
* Add bson
* Merge pull request `#4 <https://github.com/LCAS/mqtt_bridge/issues/4>`_ from Roboauto/master
  Bridge fixed not to fall when ros msg cannot be created
* bridge fixed not to fall when ros msg cannot be created
* Contributors: 5tan, Junya Hayashi, Tomas Cernik, Yuma Mihira, Yuma.M, kapilPython

0.1.6 (2017-11-10)
------------------
* Update CHANGELOG
* Merge pull request `#2 <https://github.com/LCAS/mqtt_bridge/issues/2>`_ from Roboauto/master
  fix if frequency is none
* fix if frequency is none
* Contributors: Junya Hayashi, Shin

0.1.5 (2016-12-07)
------------------
* Update CHANEGLOG
* Update url in package.xml
* Contributors: Junya Hayashi

0.1.4 (2016-12-06 15:03)
------------------------
* Update CHANGELOG.rst
* Comment out python-msgpack in package.xml
* Contributors: Junya Hayashi

0.1.3 (2016-12-06 14:56)
------------------------
* Update CHANGELOG.rst
* Comment out python-inject-pip and python-paho-mqtt-pip until it will be available
* Remove changelog from README.md
* Improve declaring python dependencies
* Contributors: Junya Hayashi

0.1.2 (2016-12-04 23:00)
------------------------
* Update CHANGELOG.rst
* Fix CMakeLists.txt and package.xml
* Update CHANGELOG.rst
* Comment out catkin_add_nosetests in CMakeLists.txt
* Contributors: Junya Hayashi

0.1.1 (2016-12-04 00:32)
------------------------
* Add CHANGELOG.rst
* Contributors: Junya Hayashi

0.1.0 (2016-12-03)
------------------
* Change author and maintainer
* Fix typo: selialize -> serialize
* Delete params under the node before loading new parameters
* Implement mqtt_private_path feature
* Remove double quotation from yaml files
* Remove leading slash from MQTT topic path
* Add config to INSTALL_DIRECTORY
* Add use_tls option in demo.launch
* Update usage
* Write usage in README.md
* Add license notes (MIT)
* Implement messagepack based selization
* Rename launch file
* Implement bridge feature
* Implement ros node and mqtt client factory
* initial commit
* Contributors: Junya Hayashi
