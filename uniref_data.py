import requests
import json

def get_uniref_cluster(cluster_id, format='json'):
    """
    Fetches a UniRef cluster from the UniProt API by its cluster ID.
    
    Parameters:
        cluster_id (str): The UniRef cluster ID (e.g., 'UniRef50_P12345').
        format (str): The response format ('json', 'xml', 'fasta', 'tsv'). Default is 'json'.
    
    Returns:
        dict or str: The cluster data as a dictionary (if json), or raw string (for fasta, xml, etc.).
    
    Raises:
        ValueError: If the cluster ID is invalid or not found.
        requests.exceptions.RequestException: For network-related errors.
    """
    base_url = "https://rest.uniprot.org/uniref/"
    url = f"{base_url}{cluster_id}"
    headers = {"Accept": f"application/{format}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json() if format == 'json' else response.text
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise ValueError(f"Cluster ID '{cluster_id}' not found.") from e
        else:
            raise
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error occurred: {e}") from e

if __name__ == "__main__":
    cluster_id = "UniRef50_P05067"
    cluster_data = get_uniref_cluster(cluster_id)
    print(cluster_data.keys())
    del cluster_data['members']
   
    cluster_data_json = json.dumps(cluster_data, indent=4)
    print(cluster_data_json)