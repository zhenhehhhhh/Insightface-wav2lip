import requests
import wget
from urllib.parse import urlencode
from urllib.parse import quote
import subprocess, os


def dytts_fun(text, audio_path, ran_str):
    url = "https://tts.daoying.tech/text2audio?"
    # text= "更年轻的身体，容得下更多元的文化、审美和价值观。有一天我终于发现，不只是我们在教你们如何生活，你们也在启发我们怎样去更好的生活。那些抱怨一代不如一代的人，应该看看你们，就像我一样。"
    text = quote(text, encoding="utf-8")
    data = {
        "lan": "zh",
        "spd": 4,
        "ctp": 1,
        "pdt": 172,
        "cuid": "cjs",
        "aue": 6,
        "per": "fs_zhangyu_cn_en",
        "tex": text
    }
    data = urlencode(data, encoding="utf-8")
    r = requests.post(url, data)
    url = url + data
    print(url)
    if os.path.exists(audio_path + ran_str + '0.wav'):
        os.remove(audio_path + ran_str + '0.wav')
    wget.download(url, out=audio_path + ran_str + '0.wav')  # 下载抠除背景
    # 裁剪
    command = 'ffmpeg -i {} -ss 00:00:00.20 -t 00:10:00.00 -ac 1 -ar 16000 -y {}'.format(audio_path + ran_str + '0.wav',
                                                                                         audio_path + ran_str + '.wav')
    subprocess.call(command, shell=True)


# text = "我，大家好，我是AI虚拟主播小文，千里智慧融媒创新创业孵化基地的孵化成果之一兼代言人。"
# text = "我，因媒而生的千里智慧融媒创新创业孵化基地包括智媒实训区、产业孵化区两大专属办公区域，并设洪崖洞创意工作室实训基地、街镇乡村振兴直播实训基地。孵化基地总面积6000平方米，创业工位50+N个，倾力打造创新创业空间、智媒空间等六大空间。"
# text = "我，高新赋能精彩，年轻不惧挑战。基地建设以“新闻+创新创业”为引领，围绕“智媒+产业”生态打造，构建“媒体+政务+产业+教学+孵化”的发展格局，重点引进文化传播、数字媒体、直播赋能、电商人才、时尚创意、招商引智等领域创新创业团队。"
# text = "我，资源聚集，活力释放；资源共享，共创辉煌。因为你的加入，到2026年年底，基地将实现入驻企业60余家，培育市级科技型企业18家，毕业企业20家，积极创建国家级科技企业孵化器。"
# text = "我，创业与时代同行，创新携梦想齐飞，千里智慧融媒创新创业孵化基地欢迎您一起孵化梦想！"
# text = "我，如果你是正在创业的个人或团体，如果你拥有一家刚起步的企业。那么请加入我们，这里有一帮与你志同道合的朋友，一起交流、一起培训、一起就业、一起创业。"
#
# audio_path = 'outputs/'
# ran_str = 'audio'
# dytts_fun(text, audio_path, ran_str)
