#!/usr/bin/env python3
"""Inspect actual exported vector fonts, size, PNG decoding and SVG text."""
from pathlib import Path
import argparse, hashlib, json, xml.etree.ElementTree as ET
from pypdf import PdfReader
from PIL import Image

NAME='n9_policy_direction_response'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    reader=PdfReader(args.figures/(NAME+'.pdf'));assert len(reader.pages)==1
    page=reader.pages[0]
    width=float(page.mediabox.width)*25.4/72
    height=float(page.mediabox.height)*25.4/72
    assert abs(width-178)<.001 and abs(height-70)<.001
    fonts=[]
    for key,reference in page['/Resources'].get('/Font',{}).items():
        font=reference.get_object();assert font.get('/Subtype')!='/Type3'
        descendant=font['/DescendantFonts'][0].get_object() if '/DescendantFonts' in font else font
        descriptor=descendant.get('/FontDescriptor')
        descriptor=descriptor.get_object() if descriptor is not None else {}
        assert any(key in descriptor for key in ['/FontFile','/FontFile2','/FontFile3'])
        name=str(font.get('/BaseFont'));assert 'STIX' in name
        fonts.append({'resource':str(key),'font':name,'subtype':str(font.get('/Subtype')),'embedded':True})
    assert fonts
    with Image.open(args.figures/(NAME+'.png')) as image:
        image.load();pixels=list(image.size);dpi=list(image.info['dpi'])
        assert min(dpi)>599
        assert abs(pixels[0]/dpi[0]*25.4-178)<.1
        assert abs(pixels[1]/dpi[1]*25.4-70)<.1
    svg=ET.parse(args.figures/(NAME+'.svg'))
    texts=list(svg.getroot().iter('{http://www.w3.org/2000/svg}text'))
    assert len(texts)>20
    assert all('STIX' in text.attrib.get('style','') for text in texts)
    content=' '.join(''.join(text.itertext()) for text in texts)
    assert 'training direction' in content and 'intermediate direction' in content
    assert 'offset 0' in content and 'offset 40000' in content
    assert 'lookup' not in content.lower() and '3.8101' not in content
    result={'name':NAME,'width_mm':width,'height_mm':height,'pages':1,
        'fonts':fonts,'png_pixels':pixels,'png_dpi':dpi,'svg_text_elements':len(texts),
        'sha256':{extension:hashlib.sha256((args.figures/(NAME+'.'+extension)).read_bytes()).hexdigest()
                  for extension in ['pdf','svg','png']}}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'status':'PASS','actual_PDF_fonts_inspected':True,
        'all_fonts_embedded_no_Type3':True,'PNG_decoded_600dpi':True,
        'displayed_single_policy_only':True,'figures':[result],'new_GPUh':0},indent=2)+'\n')
    print(json.dumps({'status':'PASS','figures':1,'actual_PDF_fonts_inspected':True}))

if __name__=='__main__':main()
