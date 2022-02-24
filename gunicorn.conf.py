import multiprocessing

bind = "127.0.0.1:5000"
workers = multiprocessing.cpu_count() * 2 + 1
# TODO: Enable logging after installing and configuring CW Logs agent
# accesslog = 'logs/request.log'
# access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" "%({X-Forwarded-For}i)s"'
# errorlog = 'logs/application.log'
