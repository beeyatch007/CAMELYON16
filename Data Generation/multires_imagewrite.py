# File paths should be given with file names and extensions in the following functions


def wsi_tissue_mask(input_img_path, output_img_path):
    import multiresolutionimageinterface as mir
    import numpy as np
    from scipy.ndimage.filters import median_filter
    from skimage.transform import resize
    reader = mir.MultiResolutionImageReader()
    image = reader.open(input_img_path)

    level_dims = image.getLevelDimensions(3)
    level_ds = image.getLevelDownsample(3)
    tile = image.getUCharPatch(0, 0, level_dims[0], level_dims[1], 3)
    tile_clipped = np.clip(tile, 1, 254)
    tile_od = -np.log(tile_clipped / 255.)
    D = median_filter(np.sum(tile_od, axis=2) / 3., size=3)
    raw_mask = (((D > 0.02 * -np.log(1/255.)) * (D < 0.98 * -np.log(1/255.))).astype("ubyte"))
    out_dims = image.getLevelDimensions(0)
    step_size = int(512. / int(level_ds))
    writer = mir.MultiResolutionImageWriter()

    writer.openFile(output_img_path)
    writer.setTileSize(512)
    writer.setCompression(mir.LZW)
    writer.setDataType(mir.UChar)
    writer.setInterpolation(mir.NearestNeighbor)
    writer.setColorType(mir.Monochrome)
    writer.writeImageInformation(out_dims[0], out_dims[1])
    for y in range(0, level_dims[1], step_size):
        for x in range(0, level_dims[0], step_size):
            write_t1 = np.zeros((step_size, step_size), dtype='ubyte')
            cur_t1 = raw_mask[y:y+step_size, x:x+step_size]
            write_t1[0 : cur_t1.shape[0], 0:cur_t1.shape[1]] = cur_t1
            res_t1 = resize(write_t1, (512,512), order=0, mode="constant", preserve_range=True).astype("ubyte")
            writer.writeBaseImagePart(res_t1.flatten())
    writer.finishImage()


def annotation_mask(xml_file_path, input_img_path, output_img_path):
    import multiresolutionimageinterface as mir
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(input_img_path)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_file_path)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    camelyon17_type_mask = True
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else ['_0', '_1', '_2']
    output_path = output_img_path
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map,
                            conversion_order)