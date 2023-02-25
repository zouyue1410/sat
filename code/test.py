
import time
from func_timeout import func_set_timeout


@func_set_timeout(5)
def timer():
    for num in range(1, 11):
        time.sleep(1)
        print(num)


timer()