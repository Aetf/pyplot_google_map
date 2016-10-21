#! /usr/bin/env python3

import requests
import re


def getLatLonViaYql(city, countryCode):
    # build the YQL query
    yqlQuery = ('select centroid.latitude,centroid.longitude '
                'from geo.places '
                'where text="{}" and country.code="{}" and '
                'placeTypeName.code="7"'
               ).format(city, countryCode)
    # build the URL
    yqlUrl = 'http://query.yahooapis.com/v1/public/yql'
    r = requests.get(yqlUrl, params={'q': yqlQuery})
    xmlData = r.text

    latStr = re.search('latitude>([0-9.-]+)<', xmlData)
    lonStr = re.search('longitude>([0-9.-]+)<', xmlData)
    if latStr is None or lonStr is None:
        print('Warning, unknown city {}'.format(city))
        return (0, 0)

    lat = float(latStr.group(1))
    lon = float(lonStr.group(1))
    return (lat, lon)
