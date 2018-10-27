import warnings
import math

import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imresize


def plot_google_map(ax=None, height=640, width=640,
                    scale=2, resize=1, mapType='roadmap',
                    alpha=1, showLabels=True, style='', language='en',
                    markers=None, refresh=True, autoAxis=False,
                    figureResizeUpdate=True, apiKey='', plotMode=True, **kwargs):
    """Plots a google map on the current axes using the Google Static Maps API
        ax       - Axis handle
        autoAxis - Not working, don't set to True
        markers  - list of markers
    """
    if ax is None:
        ax = plt.gca()
    if scale < 1 or scale > 2:
        raise ValueError('Scale must be 1 or 2')
    if height > 640:
        height = 640
    if width > 640:
        width = 640
    if markers is None:
        markers = []

    # Store paramters in axis (for auto refreshing)
    ax.pgm_data = {
        'ax': ax,
        'height': height,
        'width': width,
        'scale': scale,
        'resize': resize,
        'mapType': mapType,
        'alpha': alpha,
        'showLabels': showLabels,
        'style': style,
        'language': language,
        'markers': markers,
        'refresh': refresh,
        'autoAxis': autoAxis,
        'figureResizeUpdate': figureResizeUpdate,
        'apiKey': apiKey
    }
    ax.pgm_data.update(kwargs)

    curAxis = list(ax.axis())
    if np.max(np.abs(curAxis)) > 500:
        return

    print('curAxis previous: {}'.format(curAxis))
    # Enforce Latitude constraints of EPSG:900913
    curAxis[2] = max(curAxis[2], -85)  # ymin
    curAxis[3] = min(curAxis[3], 85)  # ymax
    # Enforce longitude constrains
    curAxis[0] = max(curAxis[0], -180)  # xmin
    if curAxis[0] > 180:
        curAxis[0] = 0
    curAxis[1] = min(curAxis[1], 180)  # xmax
    if curAxis[1] < -180:
        curAxis[1] = 0

    if curAxis == [0.0, 1.0, 0.0, 1.0]:  # probably an empty figure
        # display world map
        curAxis = [-200, 200, -85, 85]
        ax.axis(curAxis, emit=False)

    if autoAxis:
        warnings.warn('Auto axis doesn\'t work correctly now')
        # adjust current axis limit to avoid strectched maps
        xExtent, yExtent = latLonToMeters(curAxis[2:], curAxis[0:2])
        xExtent = np.asscalar(np.diff(xExtent))  # just the size of the span
        yExtent = np.asscalar(np.diff(yExtent))
        # get axes aspect ratio
        posBox = ax.get_position()
        aspect_ratio = posBox.height / posBox.width

        if xExtent * aspect_ratio > yExtent:
            centerX = np.mean(curAxis[:2])
            centerY = np.mean(curAxis[2:])
            spanX = (curAxis[1] - curAxis[0]) / 2
            spanY = (curAxis[3] - curAxis[2]) / 2

            # enlarge the Y extent
            spanY = spanY * xExtent * aspect_ratio / yExtent  # new span
            if spanY > 85:
                spanX = spanX * 85 / spanY
                spanY = spanY * 85 / spanY
            curAxis = [centerX - spanX, centerX + spanX,
                       centerY - spanY, centerY + spanY]
        elif yExtent > xExtent * aspect_ratio:
            centerX = np.mean(curAxis[:2])
            centerY = np.mean(curAxis[2:])
            spanX = (curAxis[1] - curAxis[0]) / 2
            spanY = (curAxis[3] - curAxis[2]) / 2

            # enlarge the X extent
            spanX = spanX * yExtent / (xExtent * aspect_ratio)  # new span
            if spanX > 180:
                spanY = spanY * 180 / spanX
                spanX = spanX * 180 / spanX

            curAxis = [centerX - spanX, centerX + spanX,
                       centerY - spanY, centerY + spanY]

        # Enforce Latitude constraints of EPSG:900913
        if curAxis[2] < -85:
            curAxis[2] += (-85 - curAxis[2])
            curAxis[3] += (-85 - curAxis[2])
        if curAxis[3] > 85:
            curAxis[2] += (85 - curAxis[3])
            curAxis[3] += (85 - curAxis[3])
        ax.axis(curAxis, emit=False)  # update axis as quickly as possible, before downloading new image

    print('curAxis: {}'.format(curAxis))
    # Delete previous map from plot (if exists)
    if plotMode:  # only if in plotting mode
        curChildren = ax.get_children()
        for child in curChildren:
            if hasattr(child, 'tag') and child.tag == 'gmap':
                child.remove()
                # TODO: copy callback functions

    # Calculate zoom level for current axis limits
    xExtent, yExtent = latLonToMeters(curAxis[2:], curAxis[:2])
    minResX = np.asscalar(np.diff(xExtent)) / width
    minResY = np.asscalar(np.diff(yExtent)) / height
    print('minResX: {} minResY: {}'.format(minResX, minResY))
    minRes = max([minResX, minResY])
    tileSize = 256
    initialResolution = 2 * np.pi * 6378137 / tileSize  # 156543.03392804062 for tileSize 256 pixels
    zoomlevel = np.floor(np.log2(initialResolution / minRes))
    print('Zoom level is {}'.format(np.log2(initialResolution / minRes)))

    # Enforce valid zoom levels
    if zoomlevel < 0:
        zoomlevel = 0
    if zoomlevel > 19:
        zoomlevel = 19

    # Calculate center coordinate in WGS1984
    lat = (curAxis[2] + curAxis[3]) / 2
    lon = (curAxis[0] + curAxis[1]) / 2

    # Construct query URL
    preamble = 'http://maps.googleapis.com/maps/api/staticmap'
    location = '?center={:.10f},{:.10f}'.format(lat, lon)
    zoomStr = '&zoom={:.0f}'.format(zoomlevel)
    sizeStr = '&scale={}&size={}x{}'.format(scale, width, height)
    mapTypeStr = '&maptype={}'.format(mapType)
    if apiKey:
        keyStr = '&key={}'.format(apiKey)
    else:
        keyStr = ''

    markersStr = '&markers=' + '%7C'.join(markers)

    if language:
        languageStr = '&language={}'.format(language)
    else:
        languageStr = ''

    if mapType in ['satellite', 'hybrid']:
        filename = 'tmp.jpg'
        formatStr = '&format=jpg'
    else:
        filename = 'tmp.png'
        formatStr = '&format=png'

    sensorStr = '&sensor=false'

    if not showLabels:
        if style:
            style += '|'
        style += 'feature:all|element:labels|visibility:off'
    if style:
        styleStr = '&style={}'.format(style)
    else:
        styleStr = ''

    url = preamble + location + zoomStr + sizeStr + mapTypeStr + formatStr + markersStr + languageStr + sensorStr + keyStr + styleStr

    # Get the image
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        warnings.warn("""Unable to download map from Google servers.
                      Returned error was: {}

                      Possible reasons: no network connection, quota exceeded, or some other error.
                      Consider using an API key if quota problems persist.

                      To debug, try pasting the following URL in your browser, which may result in a more informative error:
                      {}""".format(r.reason, r.url))

    r.raw.decode_content = True
    print(r.url)
    imag = mpimg.imread(r.raw)
    height, width = imag.shape[:2]

    # Resize if needed
    if resize != 1:
        print('resized')
        imag = imresize(imag, float(resize), 'bilinear')

    # Calculate a meshgrid of pixel coordinates in EPSG:900913
    centerPixelY = np.round(height / 2)
    centerPixelX = np.round(width / 2)
    centerX, centerY = latLonToMeters(lat, lon)  # center coordinates in EPSG:900913
    curResolution = initialResolution / 2**zoomlevel / scale / resize  # meters/pixel (EPSG:900913)
    xVec = centerX + (np.arange(width) - centerPixelX) * curResolution  # x vector
    yVec = centerY + (np.arange(height)[::-1] - centerPixelY) * curResolution  # y vector
    xMesh, yMesh = np.meshgrid(xVec, yVec)  # construct meshgrid

    # convert meshgrid to WGS1984
    lonMesh, latMesh = metersToLatLon(xMesh, yMesh)

    # Next, project the data into a uniform WGS1984 grid
    uniHeight = np.round(height * resize)
    uniWidth = np.round(width * resize)
    latVect = np.linspace(latMesh[0, 0], latMesh[-1, 0], uniHeight)
    lonVect = np.linspace(lonMesh[0, 0], lonMesh[0, -1], uniWidth)
    uniLonMesh, uniLatMesh = np.meshgrid(lonVect, latVect)
    uniImag = np.zeros((uniHeight, uniWidth, 3))

    uniImag = myTurboInterp2(lonMesh, latMesh, imag, uniLonMesh, uniLatMesh)

    if plotMode:  # plot map
        # display image
        ax.hold(True)
        axImg = ax.imshow(uniImag, alpha=alpha, origin='lower',
                          extent=[lonMesh[0, 0], lonMesh[0, -1],
                                  latMesh[0, 0], latMesh[-1, 0]])
        axImg.tag = 'gmap'
        # TODO: add a dummy image to allow pan/zoom out to x2 of the image extent

        # move map to bottom (so it doesn't hide previously drawn annotations)
        axImg.set_zorder(-1)

        ax.axis(curAxis, emit=False)  # restore original zoom

        # if auto-refresh mode - override zoom callback to allow automatic
        # refresh of map upon zoom actions.
        fig = ax.figure

        if refresh:
            xlim_cid = ax.callbacks.connect('xlim_changed', update_google_map)
            ylim_cid = ax.callbacks.connect('ylim_changed', update_google_map)
            ax.pgm_data['xlim_cid'] = xlim_cid
            ax.pgm_data['ylim_cid'] = ylim_cid
        else:
            if 'xlim_cid' in ax.pgm_data:
                ax.callbacks.disconnect(ax.pgm_data['xlim_cid'])
                del ax.pgm_data['xlim_cid']
            if 'ylim_cid' in ax.pgm_data:
                ax.callbacks.disconnect(ax.pgm_data['ylim_cid'])
                del ax.pgm_data['ylim_cid']

        # set callback for figure resize function, to update extents if figure
        # is streched.
        if figureResizeUpdate:
            resize_cid = fig.canvas.mpl_connect('resize_event', update_google_map_fig)
            ax.pgm_data['resize_cid'] = resize_cid
        else:
            if 'resize_cid' in ax.pgm_data:
                fig.canvas.mpl_disconnect(ax.pgm_data['resize_cid'])
                del ax.pgm_data['resize_cid']

        return axImg
    else:  # don't plot, only return map
        return (lonVect, latVect, uniImag)


# Coordinate transformation functions

def metersToLatLon(x, y):
    """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"""
    x = np.array(x)
    y = np.array(y)
    originShift = 2 * np.pi * 6378137 / 2.0  # 20037508.342789244
    lon = (x / originShift) * 180
    lat = (y / originShift) * 180
    lat = 180 / np.pi * (2 * np.arctan( np.exp( lat * np.pi / 180)) - np.pi / 2)
    return (lon, lat)


def latLonToMeters(lat, lon):
    """Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"""
    lat = np.array(lat)
    lon = np.array(lon)
    originShift = 2 * np.pi * 6378137 / 2.0  # 20037508.342789244
    x = lon * originShift / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360 )) / (np.pi / 180)
    y = y * originShift / 180
    return (x, y)


def myTurboInterp2(X, Y, Z, XI, YI):
    """An extremely fast nearest neighbour 2D interpolation, assuming both input
    and output grids consist only of squares, meaning:
    - uniform X for each column
    - uniform Y for each row"""
    XI = XI[0,:]
    X = X[0,:]
    YI = YI[:,0]
    Y = Y[:,0]

    xiPos = np.nan * np.ones(XI.shape)
    xLen = X.size
    yiPos = np.nan * np.ones(YI.shape)
    yLen = Y.size
    # find x conversion
    xPos = 0
    for idx in range(xiPos.size):
        if XI[idx] >= X[0] and XI[idx] <= X[-1]:
            while xPos < xLen - 1 and X[xPos + 1] < XI[idx]:
                xPos += 1
            diffs = np.abs(X[xPos:xPos + 2] - XI[idx])
            if diffs[0] < diffs[1]:
                xiPos[idx] = xPos
            else:
                xiPos[idx] = xPos + 1
    # find y conversion
    yPos = 0
    for idx in range(xiPos.size):
        if YI[idx] <= Y[0] and YI[idx] >= Y[-1]:
            while yPos < yLen - 1 and Y[yPos + 1] > YI[idx]:
                yPos += 1
            diffs = np.abs(Y[yPos:yPos + 2] - YI[idx])
            if diffs[0] < diffs[1]:
                yiPos[idx] = yPos
            else:
                yiPos[idx] = yPos + 1
    yiPos = yiPos.astype(int).reshape(yiPos.size, 1)
    xiPos = xiPos.astype(int).reshape(1, xiPos.size)
    return Z[yiPos, xiPos, :]


def update_google_map(ax):
    """callback function for auto-refresh"""
    if hasattr(ax, 'pgm_data'):
        plot_google_map(**ax.pgm_data)


def update_google_map_fig(event):
    """callback function for auto-refresh"""
    for ax in event.canvas.figure.axes:
        found = False
        for child in ax.get_children():
            if hasattr(child, 'tag') and child.tag == 'gmap':
                found = True
                break
        if found:
            plot_google_map(**ax.pgm_data)
