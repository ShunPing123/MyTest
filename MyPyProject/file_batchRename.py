import os  # os.listdir列表内容不带路径  glob.glob查找文件,列表中内容带有路径,按名称排序需要特殊处理

dir_path = './resource/CCD/image/'
# 依序获取目录下文件名的列表
files_list = sorted(os.listdir(dir_path))  # 使用sorted维持原顺序,否则读取顺序会打乱  # print() type() len()
# 遍历可迭代对象的各元素,并为每个元素分配一个索引,索引的起始值为1(默认0)
for index, file in enumerate(files_list, start=1):  # enumerate()返回一个包含索引和对应值的元组,每次迭代得到的元组依序赋值给i,f
    # 如果是文件夹,处理方式:跳过
    old_file_path = os.path.join(dir_path, file)
    if os.path.isdir(old_file_path):
        continue
    # 分离文件名与扩展名  fileName = os.path.splitext(file)[0]  字符串加法操作重新拼接
    suffix = os.path.splitext(file)[1]  # 获取文件扩展名ext/suffix/fileType  str:.*
    # 构造新的文件名 f''支持在字符串中使用{}  r b u
    new_file = f'{index:05d}{suffix}'  # prefix = 'ccd_'  # 设置新名称前缀 {prefix}
    # 构建源文件和目标文件的全名 路径+文件名
    src_full_path = os.path.join(dir_path, file)
    dst_full_path = os.path.join(dir_path, new_file)
    # 重命名文件,如果目录不一样会做移动的动作
    os.rename(src_full_path, dst_full_path)
