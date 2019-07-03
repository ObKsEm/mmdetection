from .voc import VOCDataset


class ShellDataset(VOCDataset):

    CLASSES = (
        '壳牌喜力 HX6 SN 合成技术发动机油 5W-30 1L',
        '壳牌喜力 HX6 SN 合成技术发动机油 5W-30 4L',
        '壳牌喜力 HX6 合成技术发动机油 5W-40 1L',
        '壳牌喜力 HX6 合成技术发动机油 5W-40 4L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-20 1L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-20 4L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-30 1L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-30 4L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-40 1L',
        '壳牌喜力 HX7 PLUS 全合成发动机油 5W-40 4L',
        '壳牌喜力 HX8 SN 全合成发动机油 0W-20 1L',
        '壳牌喜力 HX8 SN 全合成发动机油 0W-20 4L',
        '壳牌喜力 HX8 全合成发动机油 0W-30 1L',
        '壳牌喜力 HX8 全合成发动机油 0W-30 4L',
        '壳牌喜力 HX8 全合成发动机油 0W-40 1L',
        '壳牌喜力 HX8 全合成发动机油 0W-40 4L',
        '壳牌喜力 HX5 PLUS 合成技术发动机油 10W-40 1L',
        '壳牌喜力 HX5 PLUS 合成技术发动机油 10W-40 4L',
        '壳牌极净超凡喜力 SN 天然气全合成机油 0W-20 1L',
        '壳牌极净超凡喜力 SN 天然气全合成机油 0W-20 4L',
        '壳牌极净超凡喜力 ECT C2 / C3 天然气全合成机油 0W-30 1L',
        '壳牌极净超凡喜力 ECT C2 / C3 天然气全合成机油 0W-30 4L',
        '壳牌极净超凡喜力 天然气全合成机油 0W-40 1L',
        '壳牌极净超凡喜力 天然气全合成机油 0W-40 4L',
        '壳牌超凡喜力 天然气全合成机油 5W-30 1L',
        '壳牌超凡喜力 天然气全合成机油 5W-30 4L',
        '壳牌超凡喜力 天然气全合成机油 5W-40 1L',
        '壳牌超凡喜力 天然气全合成机油 5W-40 4L',
        '壳牌极净超凡喜力 RACING 天然气全合成机油 10W-60 1L',
        '壳牌极净超凡喜力 RACING 天然气全合成机油 10W-60 4L',
        '壳牌超凡喜力 ECT C3 全合成机油中超版 5W-30 1L',
        '壳牌超凡喜力 ECT C3 全合成机油中超版 5W-30 4L',
        '壳牌超凡喜力 全合成机油中超版 5W-40 1L',
        '壳牌超凡喜力 全合成机油中超版 5W-40 4L',
        '壳牌超凡喜力 SN 天然气全合成机油中超版 0W-20 1L',
        '壳牌超凡喜力 SN 天然气全合成机油中超版 0W-20 4L',
        '壳牌超凡喜力 ECT 天然气全合成机油中超版 0W-30 1L',
        '壳牌超凡喜力 ECT 天然气全合成机油中超版 0W-30 4L',
        '壳牌超凡喜力 PROFESSIONAL ABB 天然气全合成机油中超版 0W-40 1L',
        '壳牌超凡喜力 PROFESSIONAL ABB 天然气全合成机油中超版 0W-40 4L',
        '壳牌喜力城市启停 全合成发动机油 0W-30 1L',
        '壳牌喜力动力巅峰 全合成发动机油 0W-40 1L',
        '壳牌喜力混动先锋 全合成发动机油 0W-20 1L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-40 1L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-40 4L',
        '壳牌极净超凡喜力 ECT C2 / C3 天然气全合成油 0W-30 1L',
        '壳牌极净超凡喜力 ECT C2 / C3 天然气全合成油 0W-30 4L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 X 0W-30 1L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 X 0W-30 4L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-16 1L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-16 4L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-20 1L',
        '壳牌极净超凡喜力 SN PLUS 天然气全合成油 0W-20 4L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 5W-30 1L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 5W-30 4L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 5W-40 1L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 5W-40 4L',
        '壳牌超凡喜力 ETC C3 天然气全合成油 5W-30 1L',
        '壳牌超凡喜力 ETC C3 天然气全合成油 5W-30 4L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 X 0W-30 1L',
        '壳牌超凡喜力 SN PLUS 天然气全合成油 X 0W-30 4L',
        '壳牌喜力 HX5 PLUS SN 合成技术润滑油 5W-30 4L',
        '壳牌喜力 HX5 PLUS SN 合成技术润滑油 10W-40 4L',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 1L 紫',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 4L 紫',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 1L 黄',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 4L 黄',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-40 1L',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-40 4L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-20 1L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-20 4L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-30 1L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-30 4L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-40 1L',
        '壳牌喜力 HX7 PLUS SN PLUS 全合成润滑油 5W-40 4L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 0W-20 1L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 0W-20 4L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 0W-40 1L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 0W-40 4L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 5W-30 1L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 5W-30 4L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 5W-40 1L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 5W-40 4L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 X 0W-30 1L',
        '壳牌喜力 HX8 SN PLUS 先进全合成油 X 0W-30 4L',
        '壳牌爱德王子城市穿梭 全合成发动机油 10W-40 1L',
        '壳牌爱德王子动力巅峰 全合成发动机油 15W-50 1L',
        '壳牌爱的王子水平对置 全合成发动机油 5W-40 1L',
        '壳牌超凡喜力 全合成润滑油中超限量版 5W-30 1L',
        '壳牌超凡喜力 全合成润滑油中超限量版 5W-30 4L',
        '壳牌超凡喜力 全合成润滑油中超限量版 5W-40 1L',
        '壳牌超凡喜力 全合成润滑油中超限量版 5W-40 4L',
        '壳牌超凡喜力 全合成润滑油 5W-40 1L',
        '壳牌超凡喜力 全合成润滑油 5W-40 4L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-20 1L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-20 4L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-30 1L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-30 4L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-40 1L',
        '壳牌极净超凡喜力 天然气全合成润滑油 0W-40 4L',
        '壳牌极净超凡喜力 天然气全合成润滑油 10W-60 4L',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R2 EXTRA',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R2',
        '壳牌劲霸 RIMULA R2 EXTRA 桶装',
        '壳牌劲霸 RIMULA 天然气发动机润滑油 R5 NG 10W-40',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R5 E 10W-40',
        '壳牌劲霸 RIMULA 天然气发动机润滑油 R3 NG',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R3 TURBO',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R4 PLUS增强型 15W-40',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R6 LM 10W-40',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R6 M 10W-40',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R4 X',
        '壳牌劲霸 RIMULA  R2 EXTRA 桶装',
        '壳牌劲霸 RIMULA 天然气发动机润滑油 R5 NG 10W-40 桶装',
        '壳牌劲霸 RIMULA 重负荷柴油机润滑油 R5 E 10W-40 桶装',
        '壳牌劲霸 RIMULA  R3 TURBO 桶装',
        '壳牌爱德王子 AX3 摩托车润滑油 5W-30 1L',
        '壳牌爱德王子 AX3 摩托车润滑油 10W-30 1L',
        '壳牌爱德王子 AX3 摩托车润滑油 20W-50 1L',
        '壳牌爱德王子 AX3 摩托车润滑油 15W-40 1L',
        '壳牌爱德王子 AX3 重负荷摩托车发动机油 20W-50 1L',
        '壳牌爱德王子 ULTRA 全合成摩托车润滑油 5W-40 1L',
        '壳牌爱德王子 ULTRA 全合成摩托车润滑油 15W-50 1L',
        '壳牌爱德王子 AX5 合成技术摩托车润滑油 10W-40 1L',
        '壳牌爱德王子 AX7 半合成摩托车润滑油 10W-40 1L',
        '壳牌超凡喜力 ECT C3 天然气全合成机油 5W-30 1L',
        '壳牌超凡喜力 ECT C3 天然气全合成机油 5W-30 4L',
        '壳牌喜力 HX5 SN 优质多级润滑油 5W-30 1L',
        '壳牌喜力 HX5 SN 优质多级润滑油 5W-30 4L',
        '壳牌喜力 HX5 优质多级润滑油 10W-40 1L',
        '壳牌喜力 HX5 优质多级润滑油 10W-40 4L',
        '壳牌喜力 HX6 合成技术润滑油 10W-40 1L',
        '壳牌喜力 HX6 合成技术润滑油 10W-40 4L',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 1L',
        '壳牌喜力 HX6 SN 合成技术润滑油 5W-30 4L',
        '壳牌喜力 HX7 SN 合成技术润滑油 5W-30 4L',
        '壳牌喜力 HX7 合成技术润滑油 5W-40 1L',
        '壳牌喜力 HX7 合成技术润滑油 5W-40 4L',
        '壳牌喜力 HX8 全合成润滑油 0W-20 1L',
        '壳牌喜力 HX8 全合成润滑油 0W-20 4L',
        '壳牌喜力 HX8 全合成润滑油 5W-30 4L',
        '壳牌喜力 HX8 全合成润滑油 5W-40 1L',
        '壳牌喜力 HX8 全合成润滑油 5W-40 4L',
        '壳牌喜力 HX3 优质矿物润滑油 15W-40 4L',
        '壳牌全效防冻液 OAT -30℃',
        '壳牌机动车发动机冷却液 OAT -30℃',
        '壳牌机动车发动机冷却液 OAT -45℃',
        '壳牌清洗油',
        '未识别SKU'
    )