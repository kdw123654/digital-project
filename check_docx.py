import zipfile
import xml.etree.ElementTree as ET

path = r'C:\Users\kimk1\.gemini\antigravity\brain\8ffbf7a3-3949-4f2b-be81-b6c5324e22ea\.user_uploaded\media_1791351546242.docx'
with zipfile.ZipFile(path) as z:
    rels_xml = z.read('word/_rels/document.xml.rels')
    doc_xml = z.read('word/document.xml')

rels_tree = ET.fromstring(rels_xml)
rel_map = {}
for r in rels_tree:
    rel_map[r.attrib['Id']] = r.attrib['Target']

doc_tree = ET.fromstring(doc_xml)

with open('image_placement_report.txt', 'w', encoding='utf-8') as out:
    for i, p in enumerate(doc_tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')):
        texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
        txt = ''.join(texts).strip()
        blips = p.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
        if blips or txt:
            img_names = []
            for b in blips:
                r_id = b.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                img_names.append('IMG:' + str(rel_map.get(r_id, 'unknown')))
            img_joined = ' '.join(img_names)
            out.write(f'P{i}: {img_joined} | Text: {txt}\n')

print('Done writing image_placement_report.txt')
