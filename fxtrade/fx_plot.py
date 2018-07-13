import matplotlib as mpl
import socket

# クラウドで実行する場合にバックエンドを切り替える
host = socket.gethostname()
if host == 'jrirndgcp01':
    mpl.use('Agg')

import matplotlib.pyplot as plt


def plot_history_class(history, outmethod='default'):
    # 精度の履歴をプロット
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')

    if outmethod != 'file':
        plt.show()
    else:
        plt.savefig('acc_plt.png')

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'val_loss'], loc='lower right')

    if outmethod != 'file':
        plt.show()
    else:
        plt.savefig('loss_plt.png')


def plot_history_regression(history, outmethod='default'):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')

    if outmethod != 'file':
        plt.show()
    else:
        plt.savefig('acc_plt.png')

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'val_loss'], loc='lower right')

    if outmethod != 'file':
        plt.show()
    else:
        plt.savefig('loss_plt.png')
