"""Generate printable vector ArUco sheets with an explicit physical black-border size."""
import argparse
from pathlib import Path
import cv2


def generate(output,count=8,size_mm=120):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # Eight modules per side: six data modules plus one black border each side.
    for ident in range(count):
        marker=cv2.aruco.generateImageMarker(dictionary,ident,8)
        cells=[];unit=size_mm/8;left=(210-size_mm)/2;top=40
        for row in range(8):
            for col in range(8):
                if marker[row,col]==0:cells.append(f'<rect x="{left+col*unit}" y="{top+row*unit}" width="{unit}" height="{unit}"/>')
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297"><rect width="210" height="297" fill="white"/><g fill="black">'+''.join(cells)+f'</g><text x="105" y="{top+size_mm+15}" text-anchor="middle" font-size="6">ArUco 6x6/250 • ID {ident} • {size_mm} mm black border</text><text x="105" y="{top+size_mm+25}" text-anchor="middle" font-size="4">Print at 100%. Measure the black square after printing.</text></svg>'
        (output/f'marker_{ident:02}.svg').write_text(svg,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='markers');p.add_argument('--count',type=int,default=8);p.add_argument('--size-mm',type=float,default=120)
    args=p.parse_args()
    if not 1<=args.count<=250 or not 20<=args.size_mm<=170:p.error('Use 1–250 markers and size 20–170 mm')
    generate(args.output,args.count,args.size_mm)
