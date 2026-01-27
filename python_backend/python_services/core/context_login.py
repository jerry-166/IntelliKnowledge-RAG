import contextvars

# 定义上下文变量
user_id = contextvars.ContextVar('user_id')


def set_user_context(user_id):
    user_id.set(user_id)


def get_user():
    return user_id.get(None)
