# libraries used
import xml.etree.ElementTree as ET 
import os

# the main file name
OSM_FILE = "cape-town_south-africa.osm"

# the new file which will be created
SAMPLE_FILE = "sample.osm"


def get_element(osm_file, tags=('node', 'way', 'relation')):
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    # create file start
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')

    # for each 10 elements write top one
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % 50 == 0:
            output.write(ET.tostring(element, encoding='utf-8'))
    
    # write file end
    output.write('</osm>')