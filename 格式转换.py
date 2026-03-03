from moviepy.editor import VideoFileClip

def convert_gif_to_mp4(gif_path, mp4_path):
    try:
        # 加载 GIF 文件
        clip = VideoFileClip(gif_path)
        
        # 将其写入为 MP4 文件
        # codec="libx264" 是最标准的 MP4 编码，能保证最好的兼容性
        clip.write_videofile(mp4_path, codec="libx264")
        
        # 释放资源
        clip.close()
        print(f"转换成功！文件已保存至: {mp4_path}")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# --- 使用示例 ---
# 请将下面的路径替换为你自己的文件路径
input_gif = "/home/lqz27/dyx_ws/Model-FreeNN/srlnbc/safety_gym_result.gif"   
output_mp4 = "output.mp4" 

convert_gif_to_mp4(input_gif, output_mp4)