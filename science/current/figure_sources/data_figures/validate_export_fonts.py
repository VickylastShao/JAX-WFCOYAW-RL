from pathlib import Path
import json,hashlib,sys
from pypdf import PdfReader
from PIL import Image
root=Path(sys.argv[1]);out=root/'figure_sources/data_figures/PDF_EXPORT_VALIDATION.json';spec=json.loads((root/'figure_sources/FIGURE_SPECS.json').read_text())
rows=[]
for f in spec['figures']:
 p=root/'manuscript/figures'/(f['file_stem']+'.pdf');r=PdfReader(p);assert len(r.pages)==1
 pg=r.pages[0];width=float(pg.mediabox.width)*25.4/72;assert abs(width-178)<.01
 fonts=[]
 for key,ref in pg['/Resources'].get('/Font',{}).items():
  font=ref.get_object();sub=font.get('/Subtype');assert sub!='/Type3'
  if '/DescendantFonts' in font:df=font['/DescendantFonts'][0].get_object()
  else:df=font
  desc=df.get('/FontDescriptor');desc=desc.get_object() if desc is not None else {}
  embedded=any(x in desc for x in ['/FontFile','/FontFile2','/FontFile3'])
  assert embedded,(str(p),str(key))
  fonts.append({'resource':str(key),'name':str(font.get('/BaseFont')),'subtype':str(sub),'embedded':embedded})
 png=p.with_suffix('.png')
 with Image.open(png) as im:
  im.load();dpi=im.info.get('dpi');assert dpi is not None and min(dpi)>=599
  pix=list(im.size)
 rows.append({'file_stem':f['file_stem'],'width_mm':width,'pages':1,'fonts':fonts,'PNG_pixels':pix,'PNG_dpi':dpi,'PDF_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'PNG_sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
out.write_text(json.dumps({'status':'PASS','font_check':'all resources embedded, no Type3; pypdf actual inspection','image_check':'all600dpi PNG decoded','figures':rows},indent=2)+'\n');print(str(out))
