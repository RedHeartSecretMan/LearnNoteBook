# ***Selenium***

## **原理**

- **框架底层使用 *JavaScript* 模拟真实用户对浏览器进行操作。测试脚本执行时，浏览器自动按照脚本代码做出点击，输入，打开，验证等操作，就像真实用户所做的一样，从终端用户的角度测试应用程序**
- **实现浏览器兼容性测试自动化，注意，不同的浏览器上有细微的差别**
- **可使用 *Java，Python* 等多种语言编写用例脚本**

   > ***selenium 创建一个 WebDriver 实例的流程***
   > **1, *init* 入参 *options, desired_capabilities, chrome_options* 构造浏览器启动参数
   > 2, *init* 入参 *executable_path, port, service_args, service_log_path* 创建 *service*，根据这些参数直接运行 *chromedriver*
   > 3, 根据 *init* 的入参 *keep_alive* 与 *service* 的属性 *service_url* 创建一个*command_executor* (一个执行 *selenium* 命令的执行器)
   > 4, 根据 *command_executor* 和 *1* 中生成的浏览器启动参数创建一个浏览器 *session*。即向 *command_executor* 发送一个打开浏览器的命令**

## **阐释**

- ***2* 中创建 *service* 之后，已经建立了与 *chromedriver* 进程的通信，目的是为了准备好使用浏览器的控制器**
- **3 中创建 *command_executor* 之后，可通过 *command_executor* 与 *chromedriver* 进程通信，目的是为了使用浏览器的控制器**
- **4 中创建 *session* 之后，其 *session_id* 会与这个 *WebDriver* 实例进行关联，实现与 *web* 服务器通信浏览器正常 *work*，目的是为了使用浏览器**
- **chromedriver 在 selenium 中担任的是服务器角色，chromedriver 进程操作浏览器，客户端通过 command_executor 向 chromedriver 发请求**

