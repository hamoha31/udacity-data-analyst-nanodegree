import xml.etree.cElementTree as ET
import pprint
import re
import os
import codecs
import json
import re
from collections import defaultdict
from pymongo import MongoClient

# sample file
#OSM_FILE = 'sample.osm'
OSM_FILE = 'cape-town_south-africa.osm'


# this function takes a number of file size in bytes and convert it to other sizes
def convert_bytes_to_size(num):
    for x in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num = num / 1024.0

def get_file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes_to_size(file_info.st_size)
    
size = get_file_size(OSM_FILE)
print ('File size', size)



# Useing Element Tree, we will loop over the dataset and count the number of appercence of tags.
def get_tags_count(filename):
        tags_dictionary = {}
        for event, element in ET.iterparse(filename):
            if element.tag in tags_dictionary: 
                tags_dictionary[element.tag] = tags_dictionary[element.tag] + 1
            else:
                tags_dictionary[element.tag] = 1
        return tags_dictionary

pprint.pprint(get_tags_count(OSM_FILE))


# Check all elemetn and who edit them. We are using set to count users only once [not allowing duplication].
def get_users_set(filename):
    users_set = set()
    for event, element in ET.iterparse(filename):
        for item in element:
            if 'uid' in item.attrib:
                users_set.add(item.attrib['uid'])
    return users_set

users = get_users_set(OSM_FILE)
print "Total users:", len(users)



# A reguler expresion to check the last word in element name which is usally its type [street, road, etc...] 
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

# What we want to see
expected = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway", 
                         "Place", "Road", "Square", "Street", "Trail", "Circle", "Highway"]

# Mapping appriviations to expected.
mapping = {'Ave'  : 'Avenue',
           'Blvd' : 'Boulevard',
           'Dr'   : 'Drive',
           'Ln'   : 'Lane',
           'Pkwy' : 'Parkway',
           'Rd'   : 'Road',
           'Rd.'   : 'Road',
           'St'   : 'Street',
           'st'   : 'Street',
           'street' :"Street",
           'stre' :"Street",
           'stree' :"Street",
           'Ct'   : "Court",
           'Cir'  : "Circle",
           'Cr'   : "Court",
           'ave'  : 'Avenue',
           'Hwg'  : 'Highway',
           'Hwy'  : 'Highway',
           'Sq'   : "Square"}


# Take a value and check if it is not accepted, then change it.             
def audit_street(value):
    m = street_type_re.search(value)
    if m:
        street_type = m.group()
        if street_type not in expected:
            try:
                new_type = mapping[street_type]
                value = value.replace(street_type, new_type)
            except:
                return value
    return value


# This function takes a dictunary of types and the street value. 
# If its ending not in expected add it to the dictonary.
def audit_street_type(street_types_dict, value):
    m = street_type_re.search(value)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types_dict[street_type].add(value)

# This function takes a filename, loop over nodes and ways then check if it has an address with street.
# If it is, audit its type.
def audit_street_names(filename):
    dataset = open(filename, "r")
    street_types_dict = defaultdict(set)
    for event, element in ET.iterparse(dataset, events=("start",)):
        if element.tag == "node" or element.tag == "way":
            for tag in element.iter("tag"):
                if tag.attrib['k'] == "addr:street":
                    audit_street_type(street_types_dict, tag.attrib['v'])

    return street_types_dict

types = audit_street_names(OSM_FILE)
print "Sample elements  .."
pprint.pprint(next (iter (dict(types).values())))



# Names in the dataset.
expected_city_names = ['Abbotsdale', 'Athlone', 'Atlantis', 'Belgravia', 'Bellville',
     'Bergvliet', 'Blouberg', 'Blue Downs', 'Bothasig', 'Brackenfell', 'Camps Bay', 'Cape Town',
     'Capetown', 'Capricorn', 'Century City', 'Claremont', 'Constantia', 'De Tijger',
     'Delft Cape Town', 'Diep River', 'Durbanville', 'Epping', 'Filippi',
     'Foreshore, Cape Town', 'Gardens', 'Glencairn Heights', 'Goodwood', 'Gordons Bay',
     'Grassy Park', 'Green Point', 'Hout Bay', 'Hout Bay Harbour', 'Hout Bay Heights Estate',
     'Kalbaskraal', 'Kapstaden', 'Kenilworth', 'Khayelitsha', 'Killarney Gardens, Cape Town',
     'Kommetjie', 'Kuilsriver', 'Kuilsrivier', 'Lansdowne', 'Loevenstein','Maitland',
     'Makahza', 'Manenberg', 'Marina Da Gama', 'Melkbosstrand', 'Mfuleni',
     'Milnerton', 'Milnerton,  Cape Town', 'Mitchells Plain', 'Mowbray', 'Mowbray, Cape Town',
     'Muizenberg', 'Nerina Lane', 'Newlands', 'Noordhoek', 'Noordhoek, Cape Town', 'Nyanga',
     'Observatory', 'Paarden Eiland' ,'Paarl' ,'Parklands' ,'Parow','Philadelphia',
     'Pinelands','Plumstead','Pniel','Pringle Bay','Richwood','Rondebosch','Rondebosch East','Rondebosh East',
     'Rosebank','Salt River','Scarborough','Sea Point','Sea Point, Cape Town',"Simon's Town",
     'Somerset West','Sonnekuil','Steenberg','Stellenbosch','Stellenbosch Farms','Strand',
     'Strandfotein','Suider Paarl','Sybrand Park','Table View','Techno Park','Technopark',
     'Test city','Vredehoek','Vrygrond','Welgelegen 2','Welgemoed','Wellington','Woodstock',
     'Woodstock, Cape Town','Wynberg','Zonnebloem']

# Wrong names with better names.
city_names_mapping = {
         'cape Town' : "Cape Town",
         'cape town' : "Cape Town",
         'cape-town' : "Cape Town",
         'Cape town' : "Cape Town",
         'muizenberg' : "Muizenberg",
         'rylands' : "Rylands"
}


# This function takes a value of city name. 
# If the city name is in expected names, return it. Else, change it with better one.
def audit_city(value):
    if value not in expected_city_names:
        try:
            return city_names_mapping[value]
        except:
            return value
    return value

# This function takes a filename, loop over nodes and ways then check if it has an address with city name.
# If it is, audit the name and change it.
def audit_city_name(filename):
    dataset = open(filename, "r")
    cities_list = []
    for event, element in ET.iterparse(dataset, events=("start",)):
        if element.tag == "node" or element.tag == "way":
            for tag in element.iter("tag"):
                if tag.attrib['k'] == "addr:city":
                    new_tag = tag
                    new_name = audit_city(tag.attrib['v'])
                    if new_name != tag.attrib['v']:
                        print tag.attrib['v'] + " => " + new_name
                    new_tag.attrib['v'] = new_name
                    cities_list.append(new_tag)
    return cities_list

cities = audit_city_name(OSM_FILE)


# reguler exprestions to check types
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
address_regex = re.compile(r'^addr\:')
street_regex = re.compile(r'^street')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

'''
This method takes an element and check if it is a node or a way. If not skip it.
It map the XML element to an object where we can store it as JSON.
'''
def convert_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node['type'] = element.tag
        # address details of element
        address = {}

        # for each attribute in the element, parse it into object's variable
        for attribute in element.attrib:
            if attribute in CREATED:
                if 'created' not in node:
                    node['created'] = {}
                node['created'][attribute] = element.get(attribute)
            elif attribute in ['lat', 'lon']:
                continue
            else:
                node[attribute] = element.get(attribute)
                
        # store posstion cordinates if the element has lon and lat [GPS details]
        if 'lat' in element.attrib and 'lon' in element.attrib:
            node['pos'] = [float(element.get('lat')), float(element.get('lon'))]

        # for each sub element of the root one
        for sub in element:
            # parse second-level tags for ways and populate `node_refs`
            if sub.tag == 'nd':
                if 'node_refs' not in node:
                    node['node_refs'] = []
                if 'ref' in sub.attrib:
                    node['node_refs'].append(sub.get('ref'))

            # skip the sub if it has no k or v
            if sub.tag != 'tag' or 'k' not in sub.attrib or 'v' not in sub.attrib:
                continue
                
            key = sub.get('k')
            val = sub.get('v')

            # skip the key if it is not well prepared
            if problemchars.search(key):
                continue

            # if it is an address, store it in clean way
            elif address_regex.search(key):
                key = key.replace('addr:', '')
                if key == 'city':
                    address[key] = audit_city(val)
                elif key == 'street':
                    address[key] = audit_street(val)
                else:        
                    address[key] = val

            # for others
            else:
                node[key] = val
                
        # clean address and store it in the node
        if len(address) > 0:
            node['address'] = {}
            street_full = None
            street_dict = {}
            street_format = ['prefix', 'name', 'type']
            # for each key in address
            for key in address:
                val = address[key]
                if street_regex.search(key):
                    if key == 'street':
                        street_full = val
                    elif 'street:' in key:
                        street_dict[key.replace('street:', '')] = val
                else:
                    node['address'][key] = val
            # assign street_full or fallback to compile street dict
            if street_full:
                node['address']['street'] = street_full
            elif len(street_dict) > 0:
                node['address']['street'] = ' '.join([street_dict[key] for key in street_format])
        return node
    else:
        return None
    
# take the OSM file and convert it to JSON
def convert_file(filename):
    output = "{0}.json".format(filename)
    data = []
    with codecs.open(output, "w") as fw:
        for event, element in ET.iterparse(filename):
            obj = convert_element(element)
            if obj:
                data.append(obj)
                fw.write(json.dumps(obj) + "\n")
    return data

data_objects = convert_file(OSM_FILE)


# Sample of result
print "Sample object"
print data_objects[0]



# start connection
client = MongoClient('localhost:27017')
db = client['map']
# add items one by one to DB
db.map.drop()
for item in data_objects:
    db.map.insert_one(item)


# get the old file bytes and convert it to size
old_size = convert_bytes_to_size(os.path.getsize(OSM_FILE))

# get the new file bytes and convert it to size
new_size = convert_bytes_to_size(os.path.getsize(OSM_FILE + ".json"))


print ("The old file size is: {}.".format(old_size))
print ("The new file size is: {}.".format(new_size))


# total elements in DB

total_docs = db.map.find().count()
print "Total DB documants: ", total_docs


# total number of users
unique_users = len(db.map.distinct('created.user'))
print ("Total users who participated in the map is: {} users.".format(unique_users))


# count items where type = way
total_ways = db.map.find({'type':'way'}).count()

# count items where type = node
total_nodes = db.map.find({'type':'node'}).count()

print ("Total ways in the map is: {} ways.".format(total_ways))
print ("Total nodes in the map is: {} nodes.".format(total_nodes))

# Top 10 amenities
amenities = db.map.aggregate([{"$match":{"amenity":{"$exists":1}}}, {"$group":{"_id":"$amenity",
    "count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])

print "Top 10 amenities:"
pprint.pprint(list(amenities))


# Top 5 religios
religions = db.map.aggregate([{"$match":{"amenity":{"$exists":1}, "amenity":"place_of_worship"}},
                 {"$group":{"_id":"$religion", "count":{"$sum":1}}},{"$sort":{"count":-1}}, {"$limit":5}])
print "Top 5 religios:"
pprint.pprint(list(religions))


# Top 5 resturants types
places = db.map.aggregate([{"$match":{"amenity":{"$exists":1}, "amenity":"restaurant"}},
                           {"$group":{"_id":"$cuisine", "count":{"$sum":1}}},
                           {"$sort":{"count":-1}}, {"$limit":5}])
print "Top 5 resturants:"
pprint.pprint(list(places))


# Total Number of Nando's Resuturant
nandos_count = db.map.find({"$or":[ {"name": "Nando's"}, {"name": "Nandos"}]}).count()

print "Total Number of Nando's Resuturant:", nandos_count


# Number of restaurants and cafes
eating_pleaces_count = db.map.find({"$or":[ {"amenity": "cafe"}, {"amenity": "restaurant"}]}).count()
print "Number of restaurants and cafes:", eating_pleaces_count
    