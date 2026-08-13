// 临时验证脚本：验证完整识别链路（fpcalc + AcoustID + MusicBrainz）。
// 用法：cargo run --example cli -- "音频文件路径"
use audio_processor_lib::commands;
use audio_processor_lib::fingerprint;

#[tokio::main]
async fn main() {
    let path = std::env::args().nth(1).expect("请提供音频文件路径");
    // 开发期 fpcalc 位于仓库 bin 目录
    let fpcalc = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bin")
        .join("fpcalc.exe");
    let fpcalc = fpcalc.to_string_lossy().to_string();

    println!("正在生成指纹: {path}");
    let (duration, fp) = fingerprint::compute(&fpcalc, &path).expect("指纹生成失败");
    println!("时长: {duration:.2}s, 指纹长度: {}", fp.len());
    println!("指纹前 100: {}", &fp.chars().take(100).collect::<String>());

    println!("正在识别…");
    match commands::run_identify(&fpcalc, &path).await {
        Ok(info) => {
            println!("标题: {}", info.title);
            println!("艺术家: {}", info.artist);
            if let Some(a) = &info.album {
                println!("专辑: {a}");
            }
            if let Some(d) = &info.album_date {
                println!("发行日期: {d}");
            }
            println!("置信度: {:.1}%", info.confidence);
        }
        Err(e) => eprintln!("识别失败: {e}"),
    }
}
