import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, ZeroPadding2D, Reshape
import numpy as np
import time, glob, bisect

def main():
    gpu_init()
    manu_import()
    #train(trainimport())

def gpu_init():
    config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

def manu_import():
    def adddata(dats, board, answer):
        def npless(a, b):
            for i in range(15):
                for j in range(15):
                    if a[i][j] == b[i][j]: continue
                    return a[i][j] < b[i][j]
            return False
        tb, ta = [np.copy(board)], [np.copy(answer)]
        for i in range(3):
            tb.append(np.rot90(tb[i]))
            ta.append(np.rot90(ta[i]))
        for i in range(3):
            ind = 0
            if npless(tb[0], tb[1]): ind = 1
            tb = tb[:ind] + tb[ind + 1:]
            ta = ta[:ind] + ta[ind + 1:]
        l, r = 0, len(dats)
        while r - l > 1:
            mid = (l + r) // 2
            if np.array_equal(dats[mid][0], tb[0]):
                dats[mid][1] += ta[0]
                return
            if npless(dats[mid][0], tb[0]): l = mid + 1
            else: r = mid
        dats.insert(l, [board, answer])

    def get_winner(line):
        if "--" in line: return 0
        def is_invalid(x, y): return (x < 0 or x >= 15 or y < 0 or y >= 15)
        
        board = np.zeros((15, 15), int)
        for i in range(len(line)):
            pos = line[i]
            pos = [ord(pos[0]) - ord('a'), int(pos[1])]
            board[pos[0]][pos[1]] = i % 2 + 1
        
        list_dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        list_dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0: continue
                for k in range(8):
                    dx, dy = list_dx[k], list_dy[k]
                    cnt = 1
                    for l in range(1, 5):
                        if is_invalid(i + dy * l, j + dx * l): break
                        if board[i + dy * l][j + dx * l] != board[i][j]: break
                        cnt += 1
                    if cnt == 5: return board[i][j]
        return 0
    
    @tf.function
    def calc_rotNsoftmax(dats):
        db = tf.transpose(tf.reshape(dats, [-1, 2, 15, 15]), [1, 0, 2, 3])
        db, da = db[0], db[1]
        _N = tf.shape(db)[0]
        tf.print(_N, "Data")

        def body(i, db, da):
            tb, ta = tf.reshape(db[i], [-1, 15, 15]), tf.reshape(da[i], [-1, 15, 15])
            
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.reverse(tb[-1], axis = [0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            tb = tf.reshape(tf.concat([tb, tf.reshape(tf.transpose(tf.reverse(tb[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])

            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.reverse(ta[-1], axis = [0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            ta = tf.reshape(tf.concat([ta, tf.reshape(tf.transpose(tf.reverse(ta[-1], axis = [0]), [1, 0]), [-1, 15, 15])], 0), [-1, 15, 15])
            
            def body_0(j, tb, ta):
                def body_0_0(k, tb, ta):
                    if tf.reduce_all(tf.equal(tb[j], tb[k])):
                        ta = tf.reshape(tf.concat([ta[:j], tf.reshape(ta[j] + ta[k], [-1, 15, 15]), ta[j + 1:]], 0), [-1, 15, 15])
                        tb = tf.reshape(tf.concat([tb[:k], tb[k + 1:]], 0), [-1, 15, 15])
                        ta = tf.reshape(tf.concat([ta[:k], ta[k + 1:]], 0), [-1, 15, 15])
                    else: k += 1
                    return [k, tb, ta]
                k = j + 1
                _, tb, ta = tf.while_loop(
                    lambda k, tb, ta: k < tf.shape(tb)[0],
                    body_0_0,
                    [k, tb, ta],
                    [k.shape, tf.TensorShape((None, 15, 15)), tf.TensorShape((None, 15, 15))]
                )
                return [j + 1, tb, ta]
            j = tf.constant(0)
            tf.while_loop(
                lambda j, tb, ta: j < tf.shape(tb)[0],
                body_0,
                [j, tb, ta],
                [j.shape, tf.TensorShape([None, 15, 15]), tf.TensorShape((None, 15, 15))]
            )

            db = tf.reshape(
                tf.concat([db, tb[1:]], 0),
                [-1, 15, 15]
            )
            da = tf.reshape(
                tf.concat([da[:i], ta[:1], da[i + 1:], ta[1:]], 0),
                [-1, 15, 15]
            )
            maybe_print = tf.cond(tf.equal((i + 1) % 1000, 0),
                lambda: tf.print((i + 1) / _N * 100, "% done."),
                lambda: tf.group([]))
            with tf.control_dependencies([maybe_print]): return [tf.add(i, 1), db, da]
        i = tf.constant(0)
        _, db, da = tf.while_loop(
            lambda i, db, da: i < _N,
            body,
            [i, db, da],
            [i.shape, tf.TensorShape((None, 15, 15)), tf.TensorShape((None, 15, 15))],
            262144
        )
        _da = tf.cast(da, tf.float64)
        def body1(i, da, _da):
            _da = tf.reshape(
                tf.concat(
                    [
                        _da[:i],
                        tf.reshape(da[i] / tf.reduce_sum(da[i]), [-1, 15, 15])
                    ], 0),
                [-1, 15, 15]
            )
            return [i + 1, da, _da]
        i = tf.constant(0)
        _, _, _da = tf.while_loop(
            lambda i, da, _da: i < tf.shape(da)[0],
            body1,
            [i, da, _da],
            [i.shape, da.shape, tf.TensorShape((None, 15, 15))],
            262144
        )
        return tf.reshape(
            tf.concat([tf.cast(db, tf.float64), _da], 0),
            [2, -1, 15, 15]
        )

    @tf.function
    def manu_data(dats):
        dats, train_adata = tf.cast(dats[0], tf.int32), dats[1]
        train_idata = tf.zeros([0, 15, 15], tf.int32)

        def body(i, train_idata, dats):
            train_idata = tf.reshape(
                tf.concat([
                    train_idata[:i * 3, :15, :15],
                    tf.reshape(tf.cast((tf.cast(tf.square(dats[i]), tf.int32) + dats[i]) / 2, tf.int32), [-1, 15, 15]),
                    tf.reshape(tf.cast((tf.cast(tf.square(dats[i]), tf.int32) - dats[i]) / 2, tf.int32), [-1, 15, 15]),
                    tf.reshape(tf.cast(1 - tf.square(dats[i]), tf.int32), [-1, 15, 15])
                ], 0),
                [-1, 15, 15])
            maybe_print = tf.cond(tf.equal((i + 1) % 1000, 0),
                lambda: tf.print((i + 1) / tf.shape(dats)[0] * 100, "% done."),
                lambda: tf.group([]))
            with tf.control_dependencies([maybe_print]): return [tf.add(i, 1), train_idata, dats]
        i = tf.constant(0)
        _, train_idata, _ = tf.while_loop(
            lambda i, train_idata, dats: i < tf.shape(dats)[0],
            body,
            [i, train_idata, dats],
            [tf.TensorShape(()), tf.TensorShape((None, 15, 15)), dats.shape],
            262144)
        train_idata = tf.reshape(train_idata, [-1, 3, 15, 15])
        return (train_idata, train_adata)
    
    games, line = open("games.txt", "r"), ""
    dats = []
    cnt = 0
    
    print("Importing raw data...")
    while True:
        line = games.readline()
        board = np.zeros((15, 15), int)
        if not line: break
        line = line.replace("\n", "").split()
        win = get_winner(line)
        if win != 0:
            for i in range(len(line)):
                pos = line[i]
                pos = [ord(pos[0]) - ord('a'), int(pos[1])]
                if (i + win) % 2 == 1:
                    answer = np.zeros((15, 15), int)
                    answer[pos[0]][pos[1]] = 1
                    adddata(dats, board, answer)
                board *= -1
                board[pos[0]][pos[1]] = -1
        cnt += 1
        if cnt % 1000 == 0: print("Importing %02.2f%% Done."%(cnt / 1467.96))
    games.close()

    print("Converting to numpy...")
    dats = np.array(dats).reshape(-1, 2, 15, 15)
    print("Shuffling...")
    np.random.shuffle(dats)

    print("Checking rotation...")
    dats = calc_rotNsoftmax(tf.convert_to_tensor(dats))
    
    tf.print("Manufacturing train data... ")
    dats = manu_data(dats)

    print("Saving to file...")
    unit = 50000
    datcnt = (dats[0].shape[0] - 1) // unit + 1
    pw = len(str(datcnt - 1))
    started = time.time()
    for i in range(datcnt):
        np.savez_compressed("D:\\AlphaO\\train data (input)\\%0*d.npz"%(pw, i), dat = dats[0][i * unit:(i + 1) * unit].numpy())
        np.savez_compressed("D:\\AlphaO\\train data (answer)\\%0*d.npz"%(pw, i), dat = dats[1][i * unit:(i + 1) * unit].numpy())
        rem = (dats[0].shape[0] - (i + 1) * unit) / (i + 1) / unit * (time.time() - started)
        print("Saving : %d Done. - remaining time : %dm %ds"%((i + 1) * unit, rem // 60, rem % 60))
    print("All Done!")

def trainimport():
    print("importing train data...")
    unit = 50000
    t = [glob.glob("D:\\AlphaO\\train data (input)\\*.npz"), glob.glob("D:\\AlphaO\\train data (answer)\\*.npz")]
    datasets = [[], []]
    started = time.time()
    for i in range(len(t[0])):
        tmp = [np.load(t[0][i], "dat"), np.load(t[1][i], "dat")]
        for j in range(2): datasets[j].extend(list(tmp[j]["dat"].reshape(-1)))
        rem = (len(t[0]) - i - 1) / (i + 1) * (time.time() - started)
        print("%d Done. - remaining time : %ds"%((i + 1) * unit, rem))
    datasets[0] = np.array(datasets[0]).reshape(-1, 3, 15, 15).swapaxes(1, 3).swapaxes(1, 2)
    datasets[1] = np.array(datasets[1]).reshape(-1, 15, 15)
    return (datasets[0], datasets[1])

def train(datasets):
    batch_size = 8  #int(input("batch size (default = 8) : "))
    epochs = 10000  #int(input("epochs : "))
    
    train_idata, train_adata = datasets
    train_idata = train_idata.astype(float)
    train_adata = train_adata.astype(float)
    
    model = keras.Sequential([
        ZeroPadding2D(padding = (2, 2)),
        Conv2D(192, kernel_size = (5, 5), padding = "valid", activation = tf.nn.relu),
        Dropout(0.25)
    ])
    for i in range(12):
        model.add(Conv2D(192, kernel_size = (3, 3), padding = "valid", activation = tf.nn.relu))
        model.add(ZeroPadding2D(padding = (2, 2)))
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(15 * 15, activation = tf.nn.softmax))
    model.add(Reshape((15, 15)))
    model.build(input_shape = (None, 15, 15, 3))

    model.summary()

    model.compile(
        loss = "categorical_crossentropy",
        optimizer = "nadam",
        metrics = ["accuracy"]
    )

    early_stopping = EarlyStopping(monitor = "accuracy", patience = 50)
     
    history = model.fit(
        train_idata, train_adata,
        batch_size,
        epochs,
        shuffle=True,
        callbacks=[early_stopping]
    )

    loss, acc = model.evaluate(train_idata, train_adata)
     
    print("\nLoss: {}, Acc: {}".format(loss, acc))
    model.save("pn.h5")
    print("model saved to 'pn.h5'")

if __name__ == "__main__": main()