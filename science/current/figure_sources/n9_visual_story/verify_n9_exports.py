#!/usr/bin/env python3
"""Inspect exported PDF font embedding, vector dimensions and PNG/SVG files.

Uses existing pypdf/Pillow. Does not install packages or change system fonts.
"""
from pathlib import Path
import argparse, hashlib, json
from pypdf import PdfReader
from PIL import Image
import xml.etree.ElementTree as ET

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    out=[]
    for name in ['n9_direction_response','n9_feedback_matrix']:
        p=a.figures/(name+'.pdf');reader=PdfReader(p)
        assert len(reader.pages)==1
        page=reader.pages[0];width=float(page.mediabox.width)*25.4/72
        assert abs(width-178)<.001
        fonts=[]
        for key,ref in page['/Resources'].get('/Font',{}).items():
            font=ref.get_object();assert font.get('/Subtype')!='/Type3'
            descendant=font['/DescendantFonts'][0].get_object() if '/DescendantFonts' in font else font
            desc=descendant.get('/FontDescriptor');desc=desc.get_object() if desc is not None else {}
            embedded=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
            assert embedded
            fname=str(font.get('/BaseFont'))
            assert 'STIX' in fname
            fonts.append({'resource':str(key),'font':fname,'subtype':str(font.get('/Subtype')),'embedded':True})
        with Image.open(a.figures/(name+'.png')) as im:
            im.load();pixels=list(im.size);dpi=list(im.info['dpi']);assert min(dpi)>599
            assert abs(pixels[0]/dpi[0]*25.4-178)<.1
        svg=a.figures/(name+'.svg');tree=ET.parse(svg);texts=list(tree.getroot().iter('{http://www.w3.org/2000/svg}text'))
        assert len(texts)>15
        assert all('STIX' in t.attrib.get('style','') for t in texts)
        out.append({'name':name,'width_mm':width,'pages':1,'fonts':fonts,'png_pixels':pixels,
                    'png_dpi':dpi,'svg_text_elements':len(texts),
                    'sha256':{ext:hashlib.sha256((a.figures/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ['pdf','svg','png']}})
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'status':'PASS','actual_PDF_fonts_inspected':True,
        'all_fonts_embedded_no_Type3':True,'PNG_decoded_600dpi':True,'figures':out,'new_GPUh':0},indent=2)+'\n')
    print(json.dumps({'status':'PASS','figures':2,'actual_PDF_fonts_inspected':True}))

if __name__=='__main__':main()
