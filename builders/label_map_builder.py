import os
import re
import string

import tensorflow as tf

from Chinese_aster.core import label_map
from Chinese_aster.protos import label_map_pb2


def build(config):
  if not isinstance(config, label_map_pb2.LabelMap):
    raise ValueError('config not of type label_map_pb2.LabelMap')

  character_set = _build_character_set(config.character_set)
  label_map_object = label_map.LabelMap(
    character_set=character_set,
    label_offset=config.label_offset,
    unk_label=config.unk_label)
  return label_map_object

def _build_character_set(config):
  if not isinstance(config, label_map_pb2.CharacterSet):
    raise ValueError('config not of type label_map_pb2.CharacterSet')

  source_oneof = config.WhichOneof('source_oneof')
  character_set_string = None
  if source_oneof == 'text_file':
    file_path = config.text_file
    with open(file_path, 'r') as f:
      character_set_string = f.read()
    character_set = character_set_string.split('\n')
  elif source_oneof == 'text_string':
    character_set_string = config.text_string
    character_set = character_set_string.split()
  elif source_oneof == 'built_in_set':
    if config.built_in_set == label_map_pb2.CharacterSet.LOWERCASE:
      character_set = list(string.digits + string.ascii_lowercase)
      #print('LOWERCASE character_set: ', character_set)
    elif config.built_in_set == label_map_pb2.CharacterSet.ALLCASES:
      character_set = list(string.digits + string.ascii_letters)
      #print('ALLCASES character_set: ', character_set)
    elif config.built_in_set == label_map_pb2.CharacterSet.ALLCASES_SYMBOLS:
      #character_set = list(string.printable[:-6])
      all_alphabet = get_alphabet()
      character_set = get_character_set(all_alphabet)
      #print('ALLCASES_SYMBOLS character_set: ', character_set)
    else:
      raise ValueError('Unknown built_in_set')
  else:
    raise ValueError('Unknown source_oneof: {}'.format(source_oneof))

  return character_set

def get_alphabet():
  alphabetEnglish='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  alphabetfigure = '0123456789'
  alphabetcharacter = '%#&()*+,-"./:;[]_'
  all_alphabet_English = [alphabet for alphabet in alphabetEnglish + alphabetfigure + alphabetcharacter]
  # alphabetChinese = '丁万三上不丙东中丰乐乙乳事二产亨京仑付代令件价任份企传位体侠保光公共关兴其内农冷净准出分切刘则刚利别到制力加务动勇包化北区华单南卷厂厦参发口古句只可号司合同名员品售商嘉团园国圈地址坏型塔处备复天太头子学宁它安定宝实客害寸尔尺层属山岭峪州工巨市带常平年序库应度建式张录徐态怕总恒悦户执批承指捆捷提支收数整文料方无日时明易春月有期本术材板构查标格检次武毛汉江汽沃沈沪河油波注洗津洪流浙浦海涂涛润淞温港湿源溪溶漆炉炼热熔熟牌牙物特状王环班理甘生用甲电的盘目省真码研硅示祥种称程立站符第等筋签米类精级纵绑结编缘考联聚肃肋股胡腾色艺芬花苍荣菱蒙薄行表装规计订认记许论证话诺象责货质购贸资赛路车轧轻边辽达过运远途通造邮邯部郸酒酯酸采重量金钝钢铁铝销锈锌锐锡镀长门阙防阳附际限雄集雨零面鞍顺颜风首马验鱼鲅黄麻吨超高强徽贵航铜陵富鑫芜湖攀枝钒成都城殊西昌石众朝凌新疆八一广韶松德盛梅福闽亿冶罗吴鼎信镍业泉榆和县粤深惠铸泰珠裕青科技世纪诚林隆柳贺兆五普汇永洋耀芳烘横唐正滦凯义贝氏川轮毂玉田邦柱银瑞秦皇岛黎宏龙满族自治开先佰辛澳森霸紫廊坊洸家庄宣迁九线燕汀邢台陆敬前进沧涞奥宇厚经舞沙亚凤济郑阿黑襄荆群晋大展水鄂鸿湘潭涟衡管吉四现阴澄连云盐苏丹扬洲控项镔淮申萍乡余抚后英营得矿乌晟赤峰夏淄博照莱潍寿能鲁丽伦临沂齐傅烟原美锦升汾煤威星才潞陕略渝钛胜崇昆伊犁曲靖仙振杭衢元庆彩螺纹焊接低锰桥梁锅容器解坯半硬棒协议皮背坎外左右铰链丝扁铌冰箱饼角良废条缝让步图基尾系列槽插粉冲压初船双纯灯架靠弹簧道轨渗拆宽碳圆罩铬耐统退顶盖盒镁膜排矩短锻具优素氧机盗非损耗窗下敷密桩锭绞帘绳取向屈服速降肥械字亮吕韩剪侧围裙烨混橡胶激拼寄放碱简筑截形抗震拉矫拔镦墩轴直扎输送钻探胁相及窄箔栓旋模气腐蚀侯候磨极球浸磁轭瓶裂冼套烧什塑颗粒梯锑辆克兰委托矽片涤削预辊齿空两变间导杆鎳砂请确储存卸选使见书伪鉴'
  alphabetChinese = '丁万三上不丙东中丰乐乙乳事二产亨京仑付代令件价任份企传位体侠保光公共关兴其内农冷净准出分切刘则刚利别到制力加务动勇包化北区华单南卷厂厦参发口古句只可号司合同名员品售商嘉团园国圈地址坏型塔处备复天太头子学宁它安定宝实客害寸尔尺层属山岭峪州工巨市带常平年序库应度建式张录徐态怕总恒悦户执批承指捆捷提支收数整文料方无日时明易春月有期本术材板构查标格检次武毛汉江汽沃沈沪河油波注洗津洪流浙浦海涂涛润淞温港湿源溪溶漆炉炼热熔熟牌牙物特状王环班理甘生用甲电的盘目省真码研硅示祥种称程立站符第等筋签米类精级纵绑结编缘考联聚肃肋股胡腾色艺芬花苍荣菱蒙薄行表装规计订认记许论证话诺象责货质购贸资赛路车轧轻边辽达过运远途通造邮邯部郸酒酯酸采重量金钝钢铁铝销锈锌锐锡镀长门阙防阳附际限雄集雨零面鞍顺颜风首马验鱼鲅黄麻吨超高强徽贵航铜陵富鑫芜湖攀枝钒成都城殊西昌石众朝凌新疆八一广韶松德盛梅福闽亿冶罗吴鼎信镍业泉榆和县粤深惠铸泰珠裕青科技世纪诚林隆柳贺兆五普汇永洋耀芳烘横唐正滦凯义贝氏川轮毂玉田邦柱银瑞秦皇岛黎宏龙满族自治开先佰辛澳森霸紫廊坊洸家庄宣迁九线燕汀邢台陆敬前进沧涞奥宇厚经舞沙亚凤济郑阿黑襄荆群晋大展水鄂鸿湘潭涟衡管吉四现阴澄连云盐苏丹扬洲控项镔淮申萍乡余抚后英营得矿乌晟赤峰夏淄博照莱潍寿能鲁丽伦临沂齐傅烟原美锦升汾煤威星才潞陕略渝钛胜崇昆伊犁曲靖仙振杭衢元庆彩螺纹焊接低锰桥梁锅容器解坯半硬棒协议皮背坎外左右铰链丝扁铌冰箱饼角良废条缝让步图基尾系列槽插粉冲压初船双纯灯架靠弹簧道轨渗拆宽碳圆罩铬耐统退顶盖盒镁膜排矩短锻具优素氧机盗非损耗窗下敷密桩锭绞帘绳取向屈服速降肥械字亮吕韩剪侧围裙烨混橡胶激拼寄放碱简筑截形抗震拉矫拔镦墩轴直扎输送钻探胁相及窄箔栓旋模气腐蚀侯候磨极球浸磁轭瓶裂冼套烧什塑颗粒梯锑辆克兰委托矽片涤削预辊齿空两变间导杆鎳砂请确储存卸选使见书伪鉴（）辉宜拓弘硫专样小白火描述盟栖霞溧宗淀馀伟峡全红配株会社湛蒂虏伯斗然炬坤茂玛虹佛飞铭神所湾友思渤帝久骏冠洛午住来辰佳域仁如皋冯鹤竹莆聊鹏益甬廷卓细赐皆法琛冀磐志设绍投迈墨晶冈崎木浩稀土圳乾瀚锋斯迅映迪根奇睦硕萨茨屯揭欧啸欣俱炭局澎鄯善泾景师盈财驰典峨眉清翔巴息舟嘴旗岐庚昊呈智延忠旺独抬塞雅绛桂弯阜供范井茅李贷军杨泗组试入观灰淡闻喜镇心径钠贤培镜凝印政主肖厝说曹妃甸椒网异喷颖堡教斤郭晓斌瀑汤测访爵滩启魏楚铎灿创打瞬脲改性多炸效锁挤隙蛋渔详圣仓艾屿埠果反转脾庙扣牛恩何历蒋颇知卡交馒尼致憎梦曙坞钩点胎儿香意君滨百邻钨人布覆琥珀衣或拨登赵权擅撕买里静按键票琦畅滑更由贾'
  alphabetChinese = [alphabet for alphabet in alphabetChinese]
  all_alphabet = all_alphabet_English + alphabetChinese
  return all_alphabet

def get_character_set(all_alphabet):
  charset_list = []
  # 将字典编码为utf-8格式
  for i in range(len(all_alphabet)):
    charset_list.append(all_alphabet[i].encode('utf-8'))

  # print(charset_list)
  return charset_list