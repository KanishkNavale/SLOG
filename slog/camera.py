import torch


def pixel_to_camera_coordinates(uv: torch.Tensor,
                                depth: torch.Tensor,
                                intrinsic: torch.Tensor) -> torch.Tensor:
    """Projects pixel coordinates to camera coordinates: I[u, v] -> C[x, y, z]
    Args:
        uv (torch.Tensor): pixel coordinates dims (2 x 1)
        depth (torch.Tensor): depth
        intrinsic (torch.Tensor): camera intrinsic parameter
    Returns:
        torch.Tensor: camera coordinates (3 x 1)
    """
    uv1 = torch.vstack((uv, torch.ones(1, device=uv.device, dtype=uv.dtype)))
    uvz = depth * uv1

    return torch.linalg.inv(intrinsic) @ uvz


def camera_to_world_coordinates(cam_coords: torch.Tensor,
                                extrinsic: torch.Tensor) -> torch.Tensor:
    """Projects camera coordinates to world coordinates: C[x, y, z] -> W[x, y, z]
    Args:
        cam_coords (torch.Tensor): camera coordinates (3 x 1)
        extrinsic (torch.Tensor): camera extrinsic parameter
    Returns:
        torch.Tensor: world coordinates (3 x 1)
    """
    cam_coords_1 = torch.vstack((cam_coords, torch.ones(1, device=cam_coords.device, dtype=cam_coords.dtype)))
    return (extrinsic @ cam_coords_1)[:3]


def pixel_to_world_coordinates(uv: torch.Tensor,
                               depth: torch.Tensor,
                               intrinsic: torch.Tensor,
                               extrinsic: torch.Tensor) -> torch.Tensor:
    """Projects pixel coordinates to world coordinates: I[u, v] -> W[x, y, z]
    Args:
        uv (torch.Tensor): pixel coordinates dims (2 x 1)
        depth (torch.Tensor): depth
        intrinsic (torch.Tensor): camera intrinsic parameter
        extrinsic (torch.Tensor): camera extrinsic parameter
    Returns:
        torch.Tensor: world coordinates (3 x 1)
    """
    cam_coords = pixel_to_camera_coordinates(uv, depth, intrinsic)
    return camera_to_world_coordinates(cam_coords, extrinsic)


def camera_to_pixel_coordinates(cam_coords: torch.Tensor,
                                intrinsic: torch.Tensor) -> torch.Tensor:
    """Unprojects camera coordinates to pixel coordinates C[x, y, z] -> I[u, v]
    Args:
        cam_coords (torch.Tensor): camera coordinates (3 x 1)
        intrinsic (torch.Tensor): camera intrinsic parameter
    Returns:
        torch.Tensor: pixel coordinates I[u, v]
    """
    cam_coords[:2] = cam_coords[:2] / cam_coords[2]
    uvz = intrinsic @ cam_coords
    return uvz[:2]


def world_to_cam_coordinates(world_coords: torch.Tensor,
                             extrinsic: torch.Tensor) -> torch.Tensor:
    """Unprojects world coordinates to camera coordinates W[x, y, z] -> C[x, y, z]
    Args:
        world_coords (torch.Tensor): camera coordinates (3 x 1)
        extrinsic (torch.Tensor): camera intrinsic parameter
    Returns:
        torch.Tensor: camera coordinates (3 x 1)
    """
    world_coords_1 = torch.vstack((world_coords, torch.ones(1, device=world_coords.device, dtype=world_coords.dtype)))
    cam_coords_1 = torch.linalg.inv(extrinsic) @ world_coords_1
    return cam_coords_1[:3]


def world_to_pixel_coordinates(world_coords: torch.Tensor,
                               intrinsic: torch.Tensor,
                               extrinsic: torch.Tensor) -> torch.Tensor:
    """Unprojects world coordinates to pixel coordinates W[x, y, z] -> I[u, v]
    Args:
        world_coords (torch.Tensor): world coordinates (3 x 1)
        intrinsic (torch.Tensor): camera intrinsic parameter
        extrinsic (torch.Tensor): camera extrinsic parameter
    Returns:
        torch.Tensor: pixel coordinates (2 x 1)
    """
    cam_coords = world_to_cam_coordinates(world_coords, extrinsic)
    return camera_to_pixel_coordinates(cam_coords, intrinsic)
