from time import sleep
from tqdm import tqdm, trange
# from tqdm.contrib.telegram import ttgrange
from concurrent.futures import ThreadPoolExecutor

L = list(range(100))

def progresser(n):
    interval = 0.001 / (100 - n + 2)
    total = 50
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    # pbar = trange(total, desc=text, leave=True)
    # for _ in pbar:
    #     sleep(interval)
    # tqdm.write(str(pbar))
    for _ in trange(total, desc=text):
        sleep(interval)


if __name__ == '__main__':
    # with ThreadPoolExecutor(initializer=tqdm.set_lock,
    #                         initargs=(tqdm.get_lock(),)) as p:
    #     p.map(progresser, L)

    # pbar = trange(100, leave=False)
    # for n in pbar:
    #     pbar.disable = True
    #     progresser(n)
    #     pbar.disable = False
    #     tqdm.write(str(pbar))

    # with tqdm(total=100) as pbar:
    #     for n in range(100):
    #         progresser(n)
    #         pbar.update(1)

    # for n in range(100):
    #     progresser(n)

    for n in trange(100):
        progresser(n)    

    # for i in ttgrange(100):
    #     for j in trange(int(1e6), leave=False):  # should automatically be at position=1, not 0
    #         pass
