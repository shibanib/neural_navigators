def compute_functional_connectivity(session_data, regions_of_interest=None):
    """Compute functional connectivity matrix between brain regions"""
    import numpy as np
    
    # Get LFP data and brain areas
    lfp = session_data['lfp']
    brain_areas = session_data['brain_area_lfp']
    
    # If no specific regions provided, use all unique brain areas
    if regions_of_interest is None:
        regions_of_interest = np.unique(brain_areas)
    else:
        # Filter to only include regions that are present in the data
        regions_of_interest = [r for r in regions_of_interest if r in np.unique(brain_areas)]
    
    # If fewer than 2 regions, can't compute connectivity
    if len(regions_of_interest) < 2:
        print("Not enough regions to compute connectivity")
        return None, None
    
    # Initialize connectivity matrix
    n_regions = len(regions_of_interest)
    conn_matrix = np.zeros((n_regions, n_regions))
    
    # Compute mean signal for each region
    region_signals = {}
    for region in regions_of_interest:
        # Find channels for this region - fixed this line to properly handle array comparison
        channels = [i for i, area in enumerate(brain_areas) if area == region]
        if len(channels) > 0:
            # Average LFP across channels for this region
            region_signals[region] = np.mean(lfp[channels], axis=0)
    
    # Compute correlation between each pair of regions
    for i, region1 in enumerate(regions_of_interest):
        if region1 in region_signals:
            for j, region2 in enumerate(regions_of_interest):
                if region2 in region_signals:
                    # Compute Pearson correlation
                    r = np.corrcoef(region_signals[region1], region_signals[region2])[0, 1]
                    conn_matrix[i, j] = r
    
    return conn_matrix, regions_of_interest 