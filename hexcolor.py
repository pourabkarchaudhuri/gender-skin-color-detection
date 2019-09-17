

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# hex_colors = [RGB2HEX(center_colors[i]) for i in counts.keys()]

colors = [
        {
            "cluster_index": 1,
            "color": [
                224.15949675321562,
                151.35980339106652,
                132.97213203461297
            ],
            "color_percentage": 0.3111763384079546
        },
        {
            "cluster_index": 0,
            "color": [
                195.60678754068476,
                128.50474198048263,
                109.63152022315592
            ],
            "color_percentage": 0.30173866636705804
        },
        {
            "cluster_index": 4,
            "color": [
                165.48332940622336,
                105.52085296889753,
                89.78510838831329
            ],
            "color_percentage": 0.23839952811639795
        },
        {
            "cluster_index": 3,
            "color": [
                241.94566235713495,
                181.94322653176403,
                169.15645493723463
            ],
            "color_percentage": 0.07558564125610921
        },
        {
            "cluster_index": 2,
            "color": [
                107.50009606148421,
                56.94313160422364,
                49.78981748319001
            ],
            "color_percentage": 0.07309982585248019
        }
    ]



hex_colors = list()
for element in colors:
    print(element['color'])
    obtained_hexcolor = RGB2HEX(element['color'])
    hex_colors.append(obtained_hexcolor)

print(hex_colors)