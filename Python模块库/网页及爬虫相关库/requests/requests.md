
# requests简介

<font color=#888>requests基于urllib3，但是API更加好用，更加强大。

> <font color=#888>中文官方文档<https://docs.python-requests.org/zh_CN/latest/>


## 引用requests

```python
import requests
```

## 向服务器请求并获取网页


```python
#请求网页
r1=requests.get('https://www.baidu.com')
#带参数请求网页，相当于请求https://www.baidu.com/?wb=hello
r2=requests.get('https://www.baidu.com', params={'wb':'hello'})
```

requests有6种请求方式常用的get、post，以及head、put、delete、options。

> 其中POST、DELETE、PUT、GET 分别对应着"增、删、改、查"。即
>
> + POST向服务器发送数据。通常用于提交表单，提交表单后服务器会返回响应内容
> + DELETE删除服务器数据：请求删除服务器指定的文件。
> + 修改服务器数据
> + 向

+ `requests.head('https://www.baidu.com')`：只请求网页的head部分，不要网页body
+ `requests.post('https://www.baidu.com', data={'wd':'hello'})`：post发送数据 
+ `requests.put('https://temp.com/', data={'name':'lilei'})`：PUT请求向服务器端发送数据改变信息，就像数据库的update操作一样
+ `requests.delete('https://temp.com/file/a.jpg`：请求删除服务器中的某一个资源。很少见。
+ OPTIONS：极少使用。用于获取当前URL所支持的方法。若请求成功，则它会在HTTP头中包含一个名为“Allow”的头，值是所支持的方法，如“GET, POST”。

> get的params也可以用与提交表单，但是表单信息会明文在url中显示，并且url长度不能超过255个字符。

https://zhuanlan.zhihu.com/p/20410446

# 带参数的get请求
r = requests.get('https://api.github.com/events', params = {'key1': 'value1', 'key2': 'value2'}) 

# 自定义headers 
r = requests.get('https://api.github.com/some/endpoint', headers={'user-agent': 'my-app/0.0.1'}) 

# 自定义cookies
r = requests.get('http://httpbin.org/cookies', cookies=dict(cookies_are='working'))   
r = requests.get(url, cookies=requests.cookies.RequestsCookieJar().set('tasty_cookie', 'yum', 
                                                                        domain='httpbin.org', 
                                                                        path='/cookies'))
#禁用重定向                                                                    
r = requests.get('http://github.com', allow_redirects=False) 
# 设置请求超时时间
r = requests.get('http://github.com', timeout=0.001)  


# 固定写法，post请求发表单数据，自动Content-Type=application/x-www-form-urlencoded
r = requests.post('http://httpbin.org/post', data = {'key':'value'})  
r = requests.post('http://httpbin.org/post', data=(('key1', 'value1'), ('key1', 'value2'))) #多个元素同一个key

# 固定写法，post发生json请求，自动Content-Type=application/json
r = requests.post('https://api.github.com/some/endpoint', data=json.dumps({'some': 'data'}))
r = requests.post('https://api.github.com/some/endpoint', json={'some': 'data'})

# 上传文件
r = requests.post('http://httpbin.org/post', files={'file': open('report.xls', 'rb')})  
r = requests.post(url, files={'file': ('report.xls', 
                                    open('report.xls', 'rb'), 
                                    'application/vnd.ms-excel', 
                                    {'Expires': '0'})})  # 显式地设置文件名，文件类型和请求头
# 其他请求
r = requests.put('http://httpbin.org/put', data = {'key':'value'})
r = requests.delete('http://httpbin.org/delete')
r = requests.head('http://httpbin.org/get')
r = requests.options('http://httpbin.org/get')



r.status_code： 返回的状态码
r.url： 打印输出拼接后的URL
r.text：响应的body内容
r.encoding: 改变响应body的编码方式，r.text有乱码时改I
r.content: 二进制的响应body。content会自动解码 gzip 和deflate压缩
r.headers: 以字典对象存储服务器响应头，但是这个字典比较特殊，字典键不区分大小写，若键不存在则返回None
r.json(): 响应的json内容
r.raw(): 原始套接字响应
r.cookies['example_cookie_name']: 访问响应中的cookies
r.history: requests默认自动重定向，可以看重定向的记录