import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def adapt_timestamp_to_time_period(events_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Adapt the event timestamp to a day period feature

        Args:
            events: Raw data.
        Returns:
            Aggregated data by user
    """

    # 1. Conversion du timestamp
    events_dataset['timestamp'] = pd.to_datetime(events_dataset['event_time'])

    # 2. Extraction des composantes temporelles
    events_dataset['hour'] = events_dataset['timestamp'].dt.hour
    events_dataset['day'] = events_dataset['timestamp'].dt.dayofweek
    return events_dataset

def group_events_by_client(events_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Preprocesses the data for users.

    Args:
        events: Raw data.
    Returns:
        Aggregated data by user
    """

    events_dataset = adapt_timestamp_to_time_period(events_dataset)


    # 4. Encodage one-hot des types d’événements (sécurisé pour les colonnes manquantes)
    events_dataset = pd.get_dummies(events_dataset, columns=['event_type'], prefix='event_type')
    for col in ['event_type_view', 'event_type_cart', 'event_type_purchase']:
        if col not in events_dataset:
            events_dataset[col] = 0

    # 5. Agrégats principaux
    agg_behaviour = events_dataset.groupby('user_id').agg(
        total_views=('event_type_view', 'sum'),
        total_cart=('event_type_cart', 'sum'),
        total_purchase=('event_type_purchase', 'sum'),
        unique_categories=('category_id', 'nunique'),
        last_activity=('timestamp', 'max')
    ).reset_index()

    # 6. Total dépensé uniquement sur les achats
    total_spent = (
        events_dataset[events_dataset['event_type_purchase'] == 1]
        .groupby('user_id')['price']
        .sum()
        .rename('total_spent')
        .reset_index()
    )

    agg_behaviour = agg_behaviour.merge(total_spent, on='user_id', how='left')
    agg_behaviour['total_spent'] = agg_behaviour['total_spent'].fillna(0)

    # 7. KPIs dérivés
    agg_behaviour['avg_basket'] = agg_behaviour['total_spent'] / agg_behaviour['total_purchase'].replace(0, np.nan)
    agg_behaviour['conversion_rate'] = agg_behaviour['total_purchase'] / agg_behaviour['total_views'].replace(0, np.nan)

    # 8. Répartition par moments de la journée (%)
    time_dist = (
        events_dataset.groupby(['user_id', 'time_period'])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda d: d.div(d.sum(axis=1), axis=0))
    )

    # 9. Heure moyenne d’activité (cyclique)
    events_dataset['hour_rad'] = 2 * np.pi * events_dataset['hour'] / 24
    hour_avg = events_dataset.groupby('user_id').agg(
        hour_cos=('hour_rad', lambda x: np.mean(np.cos(x))),
        hour_sin=('hour_rad', lambda x: np.mean(np.sin(x)))
    ).reset_index()
    hour_avg['peak_hour'] = np.arctan2(hour_avg['hour_sin'], hour_avg['hour_cos']) * (24 / (2 * np.pi))
    hour_avg['peak_hour'] = hour_avg['peak_hour'] % 24

    # 10. Récence en jours
    now = events_dataset['timestamp'].max()
    agg_behaviour['recency_days'] = (now - agg_behaviour['last_activity']).dt.days

    # 11. Fusion finale + nettoyage
    users_dataset = (
        agg_behaviour
        .merge(time_dist, on='user_id', how='left')
        .merge(hour_avg[['user_id', 'peak_hour']], on='user_id', how='left')
        .drop(columns=['last_activity'])
        .fillna(0)
    )

    return users_dataset


def add_categories(users_dataset: pd.DataFrame, raw_dataset: pd.DataFrame, nb_categories: int) -> pd.DataFrame:
    """Add each category representing the

    Args:
        users_dataset: Users dataset.
        raw_dataset: Raw dataset
        nb_categories: Number of categories to add.
    Returns:
        Dataset with added categories.
    """

    # Nombre de catégories à garder
    TOP_N = nb_categories

    # 1. Identifier les TOP_N catégories achetées
    top_categories = (
        raw_dataset[raw_dataset['event_type_purchase'] == 1]
        .groupby('category_id')
        .size()
        .sort_values(ascending=False)
        .head(TOP_N)
        .index
    )

    # 2. Créer un pivot avec achats uniquement, autres regroupées
    purchase_pivot = (
        raw_dataset[raw_dataset['event_type_purchase'] == 1]
        .assign(category_id=lambda x: x['category_id'].where(x['category_id'].isin(top_categories), 'other'))
        .groupby(['user_id', 'category_id'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 3. Ajouter le préfixe "cat_" sauf pour 'user_id'
    purchase_pivot.columns = [
        'user_id' if col == 'user_id' else f"cat_{col}" for col in purchase_pivot.columns
    ]

    # 4. Merge avec ton dataframe final
    df_final = users_dataset.merge(purchase_pivot, on='user_id', how='left').fillna(0)

    # Optionnel : normaliser en pourcentage
    category_cols = [col for col in purchase_pivot.columns if col != 'user_id']
    df_final[category_cols] = df_final[category_cols].div(
        df_final[category_cols].sum(axis=1).replace(0, np.nan),
        axis=0
    ).fillna(0)

    return df_final

def normalise_dataset(final_dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalise all features

        Args:
            final_dataset: Final dataset before scaling.
        Returns:
            Scaled dataset.
        """
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(final_dataset)
    return scaled_dataset
