use image::{GrayImage, Luma};
use mnist::MnistBuilder;
use ndarray::Array4;
use onnxruntime::{LoggingLevel, environment::Environment, tensor::OrtOwnedTensor};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ONNX Runtime環境を作成
    let environment = Environment::builder()
        .with_log_level(LoggingLevel::Warning)
        .build()?;

    // モデルをロード
    let mut session = environment
        .new_session_builder()?
        .with_model_from_file("results/cnn_mnist_model.onnx")?;

    // MNIST データセットの読み込み
    let mnist = MnistBuilder::new()
        .base_path("data/MNIST/raw")
        .download_and_extract()
        .finalize();

    // ランダムに1つの画像を選択
    let img_len = mnist.trn_img.len();
    let mut rng = rand::thread_rng(); // ランダム数生成器
    let random_index = rng.gen_range(0..img_len / (28 * 28)); // 画像数分のインデックスをランダムに選ぶ

    // ランダムに選ばれた画像のピクセルデータを取得
    let start_index = random_index * 28 * 28; // 画像の開始インデックス
    let end_index = start_index + (28 * 28); // 画像の終了インデックス
    let image = &mnist.trn_img[start_index..end_index]; // 選択された画像のピクセルデータ

    // 画像の前処理 (28x28 の画像を 1x1x28x28 に変換)
    let input_data = Array4::from_shape_fn((1, 1, 28, 28), |(_, _, k, l)| {
        let index: usize = k * 28 + l;
        image[index] as f32 / 255.0 // 正規化
    });
    // let input_data: Array4<f32> = Array4::zeros((1, 1, 28, 28));

    // OrtOwnedTensor に変換
    let input_tensor = vec![input_data];

    // 推論を実行
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor)?;

    // 出力結果を取得
    let output_tensor = &outputs[0];

    // OrtOwnedTensor の内部データを Vec<f32> に変換
    let output_vec: Vec<f32> = match output_tensor.as_slice() {
        Some(slice) => slice.to_vec(),
        None => return Err("Failed to get slice from output tensor".into()),
    };

    // 入力画像の保存
    save_image(&image, "results/input_image.png")?;

    // 出力結果を表示（分類結果）
    let predicted_class = output_vec
        .iter()
        .cloned()
        .position(|v| v == output_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .unwrap();
    println!("Predicted class: {}", predicted_class);

    Ok(())
}

fn save_image(image_data: &[u8], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 画像データを28x28のグレースケール画像に変換
    let img = GrayImage::from_fn(28, 28, |x, y| {
        let pixel_value = image_data[(y * 28 + x) as usize] as u8;
        Luma([pixel_value])
    });

    // 画像をファイルに保存
    img.save(filename)?;
    println!("Image saved as {}", filename);

    Ok(())
}
