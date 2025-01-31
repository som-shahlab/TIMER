import argparse
from google.cloud import bigquery
import pandas as pd
from utils import setup_bigquery_client
import pandas_gbq
pandas_gbq
import os

def sample_tuples(sample_size, sampling_method, client, top_specialty_num=200, project_id, dataset_id=):
    if sampling_method == "random":
        query = f"""
        SELECT 
            person_id
        FROM 
            `{project_id}.{dataset_id}.visit_occurrence`
        GROUP BY person_id
        ORDER BY rand()
        LIMIT {sample_size}
        """
        df = pandas_gbq.read_gbq(query, project_id=client.project)
    elif sampling_method == "specialty_stratified":
        user_per_specialty = (sample_size - 1) // top_specialty_num + 1
        query = f"""
        WITH joined_data AS (
        SELECT 
            v.person_id,
            v.visit_occurrence_id,
            v.visit_concept_id,
            v.visit_start_date,
            v.visit_end_date,
            v.visit_type_concept_id,
            v.provider_id,
            p.specialty_concept_id,
            p.care_site_id,
            p.year_of_birth,
            p.gender_concept_id,
            p.provider_source_value,
            p.specialty_source_value,
            p.specialty_source_concept_id,
            p.gender_source_value,
            p.gender_source_concept_id
        FROM 
            `{project_id}.{dataset_id}.visit_occurrence` v
        LEFT JOIN 
            `{project_id}.{dataset_id}.provider` p
        ON 
            v.provider_id = p.provider_id
        ),
        top_specialty AS (
        SELECT specialty_source_value,
        COUNT(*) AS num
        FROM joined_data
        WHERE specialty_source_value is not NULL and specialty_source_value <> 'Unknown'
        Group by specialty_source_value
        ORDER BY num desc
        LIMIT {top_specialty_num}
        ),
        row_with_top_specialty AS (
        SELECT person_id,
        provider_id,
        visit_occurrence_id,
        a.specialty_source_value,
        visit_start_date,
        visit_end_date,
        ROW_NUMBER() OVER (PARTITION BY a.specialty_source_value ORDER BY rand()) AS row_index_by_specialty
        FROM joined_data a
        JOIN top_specialty b
        ON a.specialty_source_value = b.specialty_source_value
        )
        SELECT person_id,
        visit_occurrence_id,
        provider_id,
        specialty_source_value,
        visit_start_date,
        visit_end_date,
        FROM row_with_top_specialty
        WHERE row_index_by_specialty <= {user_per_specialty}
        LIMIT {sample_size}
        """
        df = pandas_gbq.read_gbq(query, project_id=client.project)

    return df


def main(args):
    client = setup_bigquery_client()
    
    print(f"Sampling {args.sample_n} patients using method: {args.sampling_method}")
    if args.sampling_method == "random":
        sampled_data = sample_tuples(args.sample_n, "random", client)
    elif args.sampling_method == "specialty_stratified":
        sampled_data = sample_tuples(args.sample_n, "specialty_stratified", client)
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    print(f"Number of sampled data: {len(sampled_data)}")
    print(f"Saving data to: {args.path_to_output_file}")
    sampled_data.to_csv(args.path_to_output_file, index=False)
    print(f"Data saved successfully. File size: {os.path.getsize(args.path_to_output_file)} bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample patients from  OMOP population.")
    parser.add_argument("--path_to_extract_or_dataset", type=str, help="BigQuery dataset path")
    parser.add_argument("--path_to_output_file", type=str)
    parser.add_argument("--sampling_method", type=str, default="random", choices=["random", "specialty_stratified"], help="Sampling method")
    parser.add_argument("--sample_n", type=int, required=True, help="Number of patients to sample")
    
    args = parser.parse_args()
    main(args)
    
    # Add this line to check if the file was created
    print(f"Output file exists: {os.path.exists(args.path_to_output_file)}")
