# 以下是对上述代码的优化，并且保存更多的平均指标，如精确率、召回率和 F1 值。优化内容包括：
# 代码结构优化：将部分功能封装成函数，提高代码的可读性和可维护性。
# 增加评价指标：使用 sklearn 库计算精确率、召回率和 F1 值，并保存到 CSV 文件中。
# 减少重复代码：将训练集和验证集的生成器配置提取到一个函数中。

import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import datetime

warnings.filterwarnings("ignore")


def create_save_dir():
    """创建以当前日期命名的文件夹"""
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    save_dir = Path(f'./{current_date}')
    save_dir.mkdir(exist_ok=True)
    return save_dir


def proc_img(filepath):
    """创建一个包含文件路径和图片标签的 DataFrame"""
    labels = [str(filepath[i]).split(os.path.sep)[-2] for i in range(len(filepath))]
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    df = pd.concat([filepath, labels], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def get_image_generators(train_df, val_df, test_df):
    """获取训练集、验证集和测试集的图像生成器"""
    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )
    val_images = train_generator.flow_from_dataframe(
        dataframe=val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    return train_images, val_images, test_images


def build_model():
    """构建模型"""
    pretrained_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False
    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(36, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_model(model, test_images):
    """评估模型并计算精确率、召回率和 F1 值"""
    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_images.classes
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    return precision, recall, f1


def main():
    start_time = datetime.datetime.now()
    save_dir = create_save_dir()

    train_dir = Path('F:/DataSetRed/1 Fruits and Vegetables Image Recognition/Fruits and Vegetables Image Recognition/train')
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))
    test_dir = Path('F:/DataSetRed/1 Fruits and Vegetables Image Recognition/Fruits and Vegetables Image Recognition/test')
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))
    val_dir = Path('F:/DataSetRed/1 Fruits and Vegetables Image Recognition/Fruits and Vegetables Image Recognition/validation')
    val_filepaths = list(val_dir.glob(r'**/*.jpg'))

    train_df = proc_img(train_filepaths)
    test_df = proc_img(test_filepaths)
    val_df = proc_img(val_filepaths)

    print('-- Training set --\n')
    print(f'Number of pictures: {train_df.shape[0]}\n')
    print(f'Number of different labels: {len(train_df.Label.unique())}\n')
    print(f'Labels: {train_df.Label.unique()}')

    df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()

    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        if i < len(df_unique):
            ax.imshow(plt.imread(df_unique.Filepath[i]))
            ax.set_title(df_unique.Label[i], fontsize=12)
        else:
            ax.axis('off')
    plt.tight_layout(pad=0.5)
    # plt.show()

    train_images, val_images, test_images = get_image_generators(train_df, val_df, test_df)
    model = build_model()

    history = model.fit(
        train_images,
        validation_data=val_images,
        batch_size=32,
        epochs=100
    )

    precision, recall, f1 = evaluate_model(model, test_images)

    history_df = pd.DataFrame(history.history)
    history_df['precision'] = precision
    history_df['recall'] = recall
    history_df['f1'] = f1
    history_df.to_csv(save_dir / 'training_metrics.csv', index=False)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics_plot.png')
    # plt.show()

    model.save(save_dir / 'best_model.h5')

    end_time = datetime.datetime.now()
    run_time = (end_time - start_time) / 60

    print(f"开始时间: {start_time}")
    print(f"结束时间: {end_time}")
    print(f"运行时间: {run_time}分钟")


if __name__ == "__main__":
    main()
