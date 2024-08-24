## Bugs
`win+r`:
* `explorer`: 文件资源管理器
* `iexplore`: edge浏览器
* `taskmgr`: 任务管理器
* `devmgmt.msc`: 设备管理器
* `cmd`: 命令行
* `powershell`: Powershell
### Microsoft Store 一直加载中
把Clash或第三方代理软件关掉就行了

### 触控板失灵
设备管理器 - 人机接口 - I2C - 卸载设备 - 重启
卸载设备后重启，Windows会重新下载该服务

### wsl安装报错
`wsl --install` 报错“没有注册表”。
首先检查相关服务打开没有。进入控制面板-程序-启用或关闭windows功能，打开：

* Windwos虚拟机监控服务平台
* 适用于Linux的Windows子系统
* 虚拟机平台

重启后再度尝试，若仍报错，去仓库 [wsl download](https://github.com/microsoft/WSL/releases) 找到最新release下的asset下载即可。

下载后直接打开wsl闪退，很正常，因为还没有下载任何发行版。
进入powershell
```bash
wsl.exe --list --online
```
选择想要下载的发行版后
```bash
wsl.exe -d <dist>
```
然后就可以正常启动wsl了。

### 找不到WLAN
重装 windows11 时必须联网，但是没有WLAN选项。
可能原因：
* 缺少驱动：
  * 查看：`win+r`或者`cmd`中输入 `devmgmt.msc`打开设备管理器，查看网络适配器下是否有无线网卡驱动
  * 解决方案：用U盘到官网下载驱动，如我的DELL G15 5520，到 [driver-site](https://www.dell.com/support/home/zh-cn/product-support/product/g-series-15-5520-laptop/drivers) 下载网卡的两个驱动程序，然后装到DELL里。
* 服务未启动：
  * 查看及解决方案：`win+r`中输入`taskmgr`启动任务管理器，点击服务选项卡，打开服务，找到WLAN相关的服务并设置为自动启动


